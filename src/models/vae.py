
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans

from src.config.config import GlobalConfig
from src.data.data_loader_collection import DataLoaderCollection
from src.models.hmm_gmm_prior import (
    estimate_transition_logits_from_paths,
    forward_log_marginal,
    gaussian_log_prob_diag,
    init_sticky_transition_logits,
    viterbi_decode,
)
from src.models.temporal_front import build_temporal_front, build_temporal_front_decoder

class ConditionalVAE(nn.Module):
    def __init__(
        self,
        data_loader: DataLoaderCollection,
        config: GlobalConfig,
        device: torch.device
    ):
        self.data_loader = data_loader
        self.global_config = config
        self.device = device
        super().__init__()
        self.__initialize_parameters()
        self.__initialize_weights()
    
    def __initialize_parameters(self):
        params = self.global_config.model.params
        self.n_subjects = self.data_loader.get_num_subjects()
        self.n_labs = self.data_loader.get_num_labs() if hasattr(self.data_loader, "get_num_labs") else 0
        
        # Dimensions (Hardcoded defaults as requested, but overridable via params)
        C, F, S, num_states = self.data_loader.get_vae_dims()
        self.S = S
        self.C = C
        self.F = F
        self.num_states = num_states

        self.feature_pipeline = getattr(self.global_config.cvae, "feature_pipeline", "fft")
        self.use_raw_cnn = self.feature_pipeline == "raw_cnn"
        self.F_time = F if self.use_raw_cnn else None
        # Story A: learn a spectral front-end from raw windows, reconstruct in that feature space.
        self.raw_cnn_recon_domain = params.get("raw_cnn_recon_domain", "spectral")
        self.raw_cnn_detach_target = bool(params.get("raw_cnn_detach_target", True))
        self.raw_cnn_fft_anchor_weight = float(params.get("raw_cnn_fft_anchor_weight", 0.0))
        self.raw_cnn_fft_anchor_decay_epochs = int(params.get("raw_cnn_fft_anchor_decay_epochs", 0))

        self.front_channels = params.get("front_channels", [32, 32])
        self.front_kernels = params.get("front_kernels", [15, 7])
        self.front_strides = params.get("front_strides", [2, 2])
        self.front_paddings = params.get("front_paddings", [7, 3])
        self.front_pool_kernel = params.get("front_pool_kernel", 2)
        
        # Convolutional config
        self.conv_channels = params.get("conv_channels", None)
        self.kernel_sizes = params.get("kernel_sizes", None)
        self.strides      = params.get("strides", None)
        self.paddings     = params.get("paddings", None)
        
        # Latent dimensions
        self.latent_dim = params.get("latent_dim", None)
        self.enc_hidden_dims = params.get("enc_hidden_dims", None) # MLP hidden layers
        self.dec_hidden_dims = params.get("dec_hidden_dims", None) # MLP hidden layers
        self.emb_dim = params.get("emb_dim", 0)
        self.decoder_only_conditioning = params.get("decoder_only_conditioning", False)
        self.conditioning_source = self.global_config.model.conditioning_source
        self.use_subject_conditioning = self.conditioning_source in ("subject", "subject_lab") and self.emb_dim > 0
        self.use_lab_conditioning = self.conditioning_source in ("lab", "subject_lab") and self.emb_dim > 0
        self.conditioning_dim = self.emb_dim * int(self.use_subject_conditioning) + self.emb_dim * int(self.use_lab_conditioning)
        self.use_encoder_conditioning = self.emb_dim > 0 and not self.decoder_only_conditioning
        self.use_decoder_conditioning = self.emb_dim > 0
        
        # Beta
        self.min_beta = params.get("min_beta", None)
        self.max_beta = params.get("max_beta", None)
        self.beta = self.min_beta
        self.beta_warmup_epochs = params.get("beta_warmup_epochs", None)
        self.beta_slowdown_epochs = params.get("beta_slowdown_epochs", None)
        self.no_beta_epochs = params.get("no_beta_epochs", None)
        self.beta_schedule = str(params.get("beta_schedule", "anneal"))
        self.beta_cyclical_period_epochs = int(params.get("beta_cyclical_period_epochs") or 0)
        
        # Prior
        self.prior = params.get("prior", "warm_gmm")  # standard | gmm | warm_gmm | hmm_gmm | warm_hmm_gmm
        self.gmm_warmup_epochs = params.get("gmm_warmup_epochs", 50)  # warm_gmm / warm_hmm_gmm
        self.gmm_warmup_initialized = False
        self.free_nats_per_dim = params.get("free_nats_per_dim", 0.02)
        self.num_gmm_states = params.get("num_gmm_states", self.num_states)

        # HMM-GMM prior (only used when prior in {"hmm_gmm", "warm_hmm_gmm"})
        self.hmm_warmup_epochs = params.get("hmm_warmup_epochs", self.gmm_warmup_epochs)
        self.hmm_transition_ramp_epochs = int(params.get("hmm_transition_ramp_epochs", 0))
        self.hmm_sticky_kappa = float(params.get("hmm_sticky_kappa", 0.9))
        self.hmm_estimate_transitions = bool(params.get("hmm_estimate_transitions", True))
        self.hmm_transitions_initialized = False
        self.hmm_transitions_initialized_at_hmm = False
        
        # Set torch random seed for reproducibility
        seed = self.global_config.seed
        torch.manual_seed(seed)

    def __initialize_weights(self):
        if self.use_subject_conditioning:
            self.subject_emb = nn.Embedding(self.n_subjects, self.emb_dim)
        if self.use_lab_conditioning:
            self.lab_emb = nn.Embedding(self.n_labs, self.emb_dim)

        self.front_end = None
        self.front_decoder = None
        self._front_lens: list[int] = []
        self.front_feature_channels: int | None = None
        self.front_feature_length: int | None = None

        if self.use_raw_cnn:
            self.front_end, self._front_lens, enc_in_ch = build_temporal_front(
                in_channels=self.C,
                input_length=self.F_time,
                channels=self.front_channels,
                kernels=self.front_kernels,
                strides=self.front_strides,
                paddings=self.front_paddings,
                pool_kernel=self.front_pool_kernel,
            )
            if self.raw_cnn_recon_domain == "time":
                self.front_decoder = build_temporal_front_decoder(
                    out_channels=self.C,
                    channels=self.front_channels,
                    kernels=self.front_kernels,
                    strides=self.front_strides,
                    paddings=self.front_paddings,
                    pool_kernel=self.front_pool_kernel,
                    front_lens=self._front_lens,
                )
            enc_in_ch = enc_in_ch
            enc_in_len = self._front_lens[-1]
            self.front_feature_channels = enc_in_ch
            self.front_feature_length = enc_in_len
        else:
            enc_in_ch = self.C
            enc_in_len = self.F
        
        # ----- Encoder CNN -----
        enc_layers = []
        in_ch = enc_in_ch
        L = enc_in_len
        self._lens = [L]  # store lengths through encoder (for decoder)
        for i, (out_ch, k, s, p) in enumerate(zip(self.conv_channels, self.kernel_sizes, self.strides, self.paddings)):
            enc_layers.append(nn.Conv1d(in_ch, out_ch, kernel_size=k, stride=s, padding=p))
            enc_layers.append(nn.LeakyReLU(0.2))
            in_ch = out_ch
            L = self._conv1d_out_len(L, k, s, p)
            self._lens.append(L)
        
        self.encoder_cnn = nn.Sequential(*enc_layers)

        # Flat dim via dummy pass (still fine)
        with torch.no_grad():
            dummy = torch.zeros(1, enc_in_ch, enc_in_len)
            out = self.encoder_cnn(dummy)
            self.flat_dim = out.view(1, -1).shape[1]
            self.conv_out_shape = out.shape[1:]  # (ch, len)

        # ----- Encoder MLP -----
        mlp = []
        prev = self.flat_dim + (self.conditioning_dim if self.use_encoder_conditioning else 0)
        for h in self.enc_hidden_dims:
            mlp += [nn.Linear(prev, h), nn.LeakyReLU(0.2)]
            prev = h
        self.encoder_mlp = nn.Sequential(*mlp)

        self.fc_mu = nn.Linear(prev, self.latent_dim)
        self.fc_logvar = nn.Linear(prev, self.latent_dim)
        
        # ----- Decoder MLP -----
        mlp = []
        prev = self.latent_dim + (self.conditioning_dim if self.use_decoder_conditioning else 0)
        for h in self.dec_hidden_dims:
            mlp += [nn.Linear(prev, h), nn.LeakyReLU(0.2)]
            prev = h
        mlp.append(nn.Linear(prev, self.flat_dim))
        self.decoder_mlp = nn.Sequential(*mlp)
        
        
        # ----- Decoder CNN (Transpose) -----
        dec_layers = []
        # reverse per-layer params
        rev_channels = list(reversed(self.conv_channels))
        rev_k = list(reversed(self.kernel_sizes))
        rev_s = list(reversed(self.strides))
        rev_p = list(reversed(self.paddings))
        
        # Start from last encoder output length
        L_in = self._lens[-1]
        in_ch = rev_channels[0]

        for i in range(len(rev_channels)):
            # target length is the length BEFORE the corresponding encoder conv
            target_L = self._lens[-2 - i]  # lens: [L0, L1, ..., Ln]

            k, s, p = rev_k[i], rev_s[i], rev_p[i]
            base = (L_in - 1) * s - 2 * p + k
            out_pad = target_L - base
            if not (0 <= out_pad < s):
                raise ValueError(
                    f"Cannot match target length in deconv layer {i}: "
                    f"target={target_L}, base={base}, stride={s} -> output_padding={out_pad}"
                )

            out_ch = enc_in_ch if i == len(rev_channels) - 1 else rev_channels[i + 1]
            dec_layers.append(nn.ConvTranspose1d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, output_padding=out_pad))
            if not (i == len(rev_channels) - 1):
                dec_layers.append(nn.LeakyReLU(0.2))
            L_in = target_L
            in_ch = out_ch
        self.decoder_cnn = nn.Sequential(*dec_layers)
        
        
        # GMM prior parameters (shared by GMM and HMM-GMM priors).
        # `prior_logits` doubles as mixture weights (GMM) or initial state log-probs (HMM).
        if self.prior in ("gmm", "warm_gmm", "hmm_gmm", "warm_hmm_gmm"):
            self.prior_logits = nn.Parameter(torch.zeros(self.num_gmm_states, device=self.device))
            self.prior_means = nn.Parameter(torch.randn(self.num_gmm_states, self.latent_dim))
            self.prior_logvars = nn.Parameter(torch.zeros(self.num_gmm_states, self.latent_dim, device=self.device))

        # HMM-GMM only: transition matrix logits.
        if self.prior in ("hmm_gmm", "warm_hmm_gmm"):
            self.prior_transition_logits = nn.Parameter(
                init_sticky_transition_logits(
                    self.num_gmm_states,
                    self.hmm_sticky_kappa,
                    device=self.device,
                )
            )

        self.to(self.device)

    def reset(self):
        self.__initialize_weights()
        
    def _conv1d_out_len(self, L, k, s, p, d=1):
        return (L + 2*p - d*(k-1) - 1) // s + 1

    # ---------- Core subroutines ----------
    
    def _flatten_conditioning_ids(self, conditioning_ids, index: int | None = None):
        ids = conditioning_ids
        if index is not None:
            if ids.dim() < 2 or ids.shape[-1] <= index:
                raise ValueError(
                    f"Expected conditioning_ids with last dim >= {index + 1} for conditioning_source='{self.conditioning_source}', got shape {tuple(ids.shape)}"
                )
            ids = ids[..., index]

        if ids.dim() == 1:
            return ids.repeat_interleave(self.S)
        if ids.dim() == 2:
            return ids.view(-1)
        raise ValueError(f"Unsupported conditioning_ids shape: {tuple(ids.shape)}")

    def get_conditioning_embedding(self, conditioning_ids):
        """
        conditioning_ids: (B,), (B, S) or (B, S, 2) for subject_lab
        returns: emb with shape (B*S, emb_dim)
        """
        if self.conditioning_dim == 0:
            return None

        emb_parts = []
        if self.use_subject_conditioning:
            subject_ids = self._flatten_conditioning_ids(conditioning_ids, index=0 if self.conditioning_source == "subject_lab" else None)
            emb_parts.append(self.subject_emb(subject_ids))
        if self.use_lab_conditioning:
            lab_ids = self._flatten_conditioning_ids(conditioning_ids, index=1 if self.conditioning_source == "subject_lab" else None)
            emb_parts.append(self.lab_emb(lab_ids))

        return emb_parts[0] if len(emb_parts) == 1 else torch.cat(emb_parts, dim=-1)
    
    def encode(self, x, subject_ids):
        """
        x: (B, S, C, F)
        subject_ids: (B,) or (B, S)
        returns: mu, logvar both (B, S, L)
        """
        B, S, C, F = x.shape
        
        # Get subject embeddings
        emb = self.get_conditioning_embedding(subject_ids) if self.use_encoder_conditioning else None
        
        # Flatten batch and sequence for independent processing
        x_flat = x.view(B * S, C, F)  # (B*S, C, F) — F is time length when raw_cnn

        if self.use_raw_cnn:
            x_flat = self.front_end(x_flat)

        h_conv = self.encoder_cnn(x_flat)  # (B*S, last_ch, last_len)
        h_flat = h_conv.view(B * S, -1)
        
        # Add subject embedding
        if self.use_encoder_conditioning:
            h_flat = torch.cat([h_flat, emb], dim=-1)
        
        # MLP
        h = self.encoder_mlp(h_flat)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        # Reshape back to (B, S, L)
        mu = mu.view(B, S, -1)
        logvar = logvar.view(B, S, -1)
        
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """
        Standard VAE reparameterization.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, subject_ids):
        """
        z: (B, S, L)
        subject_ids: (B,) or (B, S)
        returns: x_recon with shape (B, S, C, F)
        """
        B, S, L = z.shape
        
        # Get subject embeddings
        emb = self.get_conditioning_embedding(subject_ids) if self.use_decoder_conditioning else None
        
        z_flat = z.view(B * S, L)
        
        if self.use_decoder_conditioning:
            h_in = torch.cat([z_flat, emb], dim=-1)
        else:
            h_in = z_flat
        
        # MLP
        h_flat = self.decoder_mlp(h_in)
        
        # Reshape for CNN
        h_conv_in = h_flat.view(B * S, *self.conv_out_shape)
        
        x_recon_flat = self.decoder_cnn(h_conv_in)

        if self.use_raw_cnn and self.raw_cnn_recon_domain == "time":
            x_recon_flat = self.front_decoder(x_recon_flat)

        if self.use_raw_cnn and self.raw_cnn_recon_domain == "spectral":
            out_C = self.front_feature_channels
            out_F = self.front_feature_length
        else:
            out_C = self.C
            out_F = self.F_time if self.use_raw_cnn else self.F
        x_recon = x_recon_flat.view(B, S, out_C, out_F)

        return x_recon
    
    def forward(self, x, subject_ids):
        """
        Full VAE pass:
        - encode → mu, logvar
        - reparam → z
        - decode(z, subject_ids) → x_recon

        Returns:
            x_recon: (B, S, C, F)
            mu:      (B, S, L)
            logvar:  (B, S, L)
            z:       (B, S, L)
        """
        mu, logvar = self.encode(x, subject_ids)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z, subject_ids)
        return x_recon, mu, logvar, z
    
    def calculate_loss(self, x, x_recon, mu, logvar, z, epoch):
        """
        Compute VAE loss: reconstruction + KL divergence.
        """
        # Reconstruction loss (MSE)
        if self.use_raw_cnn and self.raw_cnn_recon_domain == "spectral":
            B, S, C, F_time = x.shape
            x_flat = x.view(B * S, C, F_time)
            target_spec = self.front_end(x_flat)
            target_spec = target_spec.view(
                B, S, self.front_feature_channels, self.front_feature_length
            )
            if self.raw_cnn_detach_target:
                target_spec = target_spec.detach()
            recon_loss = F.mse_loss(x_recon, target_spec, reduction='mean')
            if self.raw_cnn_fft_anchor_weight > 0.0:
                with torch.no_grad():
                    fft_power = torch.log1p(torch.abs(torch.fft.rfft(x, dim=-1)).pow(2))
                    fft_power = fft_power.mean(dim=2)
                    fft_power = F.interpolate(
                        fft_power.reshape(B * S, 1, -1),
                        size=self.front_feature_length,
                        mode="linear",
                        align_corners=False,
                    ).reshape(B, S, self.front_feature_length)
                recon_spec_mean = x_recon.mean(dim=2)
                anchor_loss = F.mse_loss(recon_spec_mean, fft_power, reduction='mean')
                anchor_w = self.raw_cnn_fft_anchor_weight
                if self.raw_cnn_fft_anchor_decay_epochs > 0:
                    decay = max(
                        0.0,
                        1.0 - (float(epoch) / float(self.raw_cnn_fft_anchor_decay_epochs)),
                    )
                    anchor_w *= decay
                recon_loss = recon_loss + anchor_w * anchor_loss
        else:
            recon_loss = F.mse_loss(x_recon, x, reduction='mean')
        # Regularization loss
        if self.prior == "gmm":
            reg_loss = self.gmm_prior(logvar, mu, z)
        elif self.prior == "standard":
            reg_loss = self.standard_prior(logvar, mu)
        elif self.prior == "warm_gmm":
            reg_loss = self.warm_gmm_prior(logvar, mu, z, epoch)
        elif self.prior == "hmm_gmm":
            reg_loss = self.hmm_gmm_prior(logvar, mu, z)
        elif self.prior == "warm_hmm_gmm":
            reg_loss = self.warm_hmm_gmm_prior(logvar, mu, z, epoch)
        else:
            raise ValueError(f"Unknown prior '{self.prior}'.")
        reg_loss = self.regularization_loss(reg_loss, epoch)
        return recon_loss, reg_loss
    
    def warm_gmm_prior(self, logvar: torch.Tensor, mu: torch.Tensor, z: torch.Tensor, epoch: int) -> torch.Tensor:
        if epoch < self.gmm_warmup_epochs:
            return self.standard_prior(logvar, mu)
        elif not self.gmm_warmup_initialized:
            means, logvars, logits = self.__get_kmeans_centroids()
            with torch.no_grad():
                self.prior_means.copy_(means)
                self.prior_logvars.copy_(logvars)
                self.prior_logits.copy_(logits)
            self.gmm_warmup_initialized = True
        return self.gmm_prior(logvar, mu, z)

    # ------------------------------------------------------------------ #
    # HMM-GMM prior (temporal)                                           #
    # ------------------------------------------------------------------ #

    def _log_q_per_step(self, mu: torch.Tensor, logvar: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """log q(z_t | x_t) for diagonal Gaussian encoder. Returns (B, T)."""
        return -0.5 * (
            math.log(2 * math.pi) + logvar + (z - mu).pow(2) / logvar.exp()
        ).sum(dim=-1)

    def _gmm_log_p_per_step(self, z: torch.Tensor) -> torch.Tensor:
        """log p_GMM(z_t) = log sum_k pi_k N(z_t | mu_k, sigma_k). Returns (B, T)."""
        log_emit = gaussian_log_prob_diag(z, self.prior_means, self.prior_logvars)  # (B,T,K)
        log_mix = F.log_softmax(self.prior_logits, dim=0).view(1, 1, -1)
        return torch.logsumexp(log_emit + log_mix, dim=-1)

    def _kl_with_free_bits(self, kl_per_seq: torch.Tensor, T: int) -> torch.Tensor:
        """Free-bits on time-averaged sequence KL (same per-step scale as ``gmm_prior``)."""
        kl_mean = kl_per_seq / float(T)
        free_nats_total = float(self.free_nats_per_dim) * self.latent_dim
        return torch.clamp(kl_mean, min=free_nats_total).mean()

    def hmm_gmm_prior(self, logvar: torch.Tensor, mu: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """KL with HMM-GMM sequence prior: KL[q || p_HMM(z_{1:T})].

        z, mu, logvar shapes: (B, T, L). Returns scalar.
        """
        if z.dim() != 3:
            raise ValueError(f"hmm_gmm_prior expects z of shape (B,T,L); got {tuple(z.shape)}")
        B, T, _ = z.shape

        log_q = self._log_q_per_step(mu, logvar, z).sum(dim=-1)              # (B,)
        log_emit = gaussian_log_prob_diag(z, self.prior_means, self.prior_logvars)
        log_pi = F.log_softmax(self.prior_logits, dim=0)
        log_A = F.log_softmax(self.prior_transition_logits, dim=-1)
        log_p_seq = forward_log_marginal(log_emit, log_pi, log_A)             # (B,)

        kl_per_seq = log_q - log_p_seq                                        # (B,)
        return self._kl_with_free_bits(kl_per_seq, T)

    def gmm_sequence_prior(self, logvar: torch.Tensor, mu: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """KL with i.i.d. GMM prior summed over time (same scale as HMM prior)."""
        if z.dim() != 3:
            raise ValueError(f"gmm_sequence_prior expects z of shape (B,T,L); got {tuple(z.shape)}")
        B, T, _ = z.shape

        log_q = self._log_q_per_step(mu, logvar, z).sum(dim=-1)              # (B,)
        log_p = self._gmm_log_p_per_step(z).sum(dim=-1)                       # (B,)
        kl_per_seq = log_q - log_p
        return self._kl_with_free_bits(kl_per_seq, T)

    def _safe_hmm_gmm_prior(self, logvar: torch.Tensor, mu: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """HMM-GMM KL; fall back to GMM-seq KL if numerics blow up (e.g. at HMM activation)."""
        reg = self.hmm_gmm_prior(logvar, mu, z)
        if torch.isfinite(reg):
            return reg
        return self.gmm_sequence_prior(logvar, mu, z)

    def warm_hmm_gmm_prior(
        self,
        logvar: torch.Tensor,
        mu: torch.Tensor,
        z: torch.Tensor,
        epoch: int,
    ) -> torch.Tensor:
        """Warm schedule: standard -> KMeans GMM init -> GMM seq -> HMM-GMM (optional ramp)."""
        # Phase 1: standard Gaussian KL while encoder warms up.
        if epoch < self.gmm_warmup_epochs:
            return self.standard_prior(logvar, mu)

        # Phase boundary: initialise GMM emissions (and optionally A) once.
        if not self.gmm_warmup_initialized:
            means, logvars, logits = self.__get_kmeans_centroids()
            with torch.no_grad():
                self.prior_means.copy_(means)
                self.prior_logvars.copy_(logvars)
                self.prior_logits.copy_(logits)
            self.gmm_warmup_initialized = True

        # Phase 2: i.i.d. GMM over time (sequence-scaled to match later phase).
        if epoch < self.hmm_warmup_epochs:
            return self.gmm_sequence_prior(logvar, mu, z)

        # Estimate transitions from current encoder means when HMM phase starts.
        if (
            self.hmm_estimate_transitions
            and not self.hmm_transitions_initialized_at_hmm
        ):
            self.__init_hmm_transitions()
            self.hmm_transitions_initialized_at_hmm = True
            self.hmm_transitions_initialized = True

        # Phase 3: HMM-GMM, optionally blended with GMM-seq to ease the transition.
        if self.hmm_transition_ramp_epochs > 0:
            ramp_progress = (epoch - self.hmm_warmup_epochs) / float(self.hmm_transition_ramp_epochs)
            alpha = max(0.0, min(1.0, ramp_progress))
        else:
            alpha = 1.0

        if alpha >= 1.0:
            return self._safe_hmm_gmm_prior(logvar, mu, z)
        if alpha <= 0.0:
            return self.gmm_sequence_prior(logvar, mu, z)

        hmm_reg = self._safe_hmm_gmm_prior(logvar, mu, z)
        gmm_reg = self.gmm_sequence_prior(logvar, mu, z)
        return (1.0 - alpha) * gmm_reg + alpha * hmm_reg

    @torch.no_grad()
    def __init_hmm_transitions(self, max_points: int = 200_000):
        """Estimate (pi, A) from Viterbi-like argmax assignments on encoder mu sequences."""
        was_training = self.training
        self.eval()

        x, _, subject_ids = self.data_loader.get_all_data()
        x = x.to(self.device)
        mu, _ = self.encode(x, subject_ids=subject_ids)  # (B, T, L)

        # Per-step argmax under current GMM emissions (ignores transitions for init).
        log_emit = gaussian_log_prob_diag(mu, self.prior_means, self.prior_logvars)
        log_mix = F.log_softmax(self.prior_logits, dim=0).view(1, 1, -1)
        assign = (log_emit + log_mix).argmax(dim=-1)  # (B, T)

        if assign.shape[1] < 2:
            # Sequence length 1: fall back to sticky init (no transitions to estimate).
            log_A = init_sticky_transition_logits(
                self.num_gmm_states, self.hmm_sticky_kappa, device=self.device
            )
            self.prior_transition_logits.copy_(log_A)
            self.train(was_training)
            return

        log_pi_est, log_A_est = estimate_transition_logits_from_paths(assign, self.num_gmm_states)
        log_A_sticky = init_sticky_transition_logits(
            self.num_gmm_states, self.hmm_sticky_kappa, device=self.device
        )
        # Blend empirical transitions with sticky prior to avoid collapsed A at activation.
        A_est = torch.exp(log_A_est)
        A_sticky = torch.exp(log_A_sticky)
        A_blend = 0.5 * A_est + 0.5 * A_sticky
        A_blend = A_blend / A_blend.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        self.prior_logits.copy_(log_pi_est)
        self.prior_transition_logits.copy_(A_blend.clamp_min(1e-12).log())
        self.train(was_training)
    
    def __get_kmeans_centroids(self, max_points: int = 200_000):
        """
        Fits KMeans on encoder means (mu) and returns:
        centroids: (K, D) tensor on self.device
        logvars:   (K, D) diag log-variances estimated from assigned points
        logits:    (K,)   log mixture weights from cluster frequencies
        """

        was_training = self.training
        self.eval()

        # Get all data and encode to latent means
        x, _, subject_ids = self.data_loader.get_all_data()
        x = x.to(self.device)

        with torch.no_grad():
            mu, _ = self.encode(x, subject_ids=subject_ids)                 # (B, S, D)
            Z = mu.reshape(-1, mu.size(-1))        # (N, D)

        # Optional subsample
        if max_points is not None and Z.size(0) > max_points:
            idx = torch.randperm(Z.size(0), device=Z.device)[:max_points]
            Z = Z[idx]

        # Run sklearn KMeans on CPU numpy
        Z_np = Z.detach().cpu().numpy()
        km = KMeans(
            n_clusters=self.num_gmm_states,
            n_init="auto",
            random_state=self.global_config.seed,
        )
        labels = km.fit_predict(Z_np)
        centroids_np = km.cluster_centers_

        # Back to torch on device
        centroids = torch.tensor(centroids_np, device=self.device, dtype=Z.dtype)

        # Mixture weights -> logits
        counts = np.bincount(labels, minlength=self.num_gmm_states).astype(np.float64)
        probs = counts / (counts.sum() + 1e-12)
        logits = torch.tensor(np.log(probs + 1e-12), device=self.device, dtype=Z.dtype)
        
        # Per-cluster diagonal variances -> logvars
        logvars = torch.zeros(self.num_gmm_states, Z.size(-1), device=self.device, dtype=Z.dtype)
        labels_t = torch.tensor(labels, device=self.device)
        
        var_floor = 1e-2

        for k in range(self.num_gmm_states):
            mask = labels_t == k
            if mask.any():
                diff = Z[mask] - centroids[k]
                var = diff.pow(2).mean(dim=0).clamp_min(var_floor)
                logvars[k] = var.log()
            else:
                logvars[k] = torch.zeros(Z.size(-1), device=self.device, dtype=Z.dtype)

        self.train(was_training)
        return centroids, logvars, logits
    
    
    def gmm_prior(self, logvar: torch.Tensor, mu: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # Flatten batch and sequence dimensions: (B*S, L)
        z = z.view(-1, self.latent_dim)
        mu = mu.view(-1, self.latent_dim)
        logvar = logvar.view(-1, self.latent_dim)
        
        # 1. log q(z|x) ~ log N(z | mu, exp(logvar))
        # Constant term -0.5*L*log(2pi) cancels out in KL, but keeping for completeness
        # Sum over latent dimension L
        log_q_z_x = -0.5 * (math.log(2 * math.pi) + logvar + (z - mu).pow(2) / logvar.exp()).sum(dim=1)
        
        # 2. log p(z) ~ log sum_k pi_k N(z | mu_k, exp(logvar_k))
        # Prior parameters
        prior_means = self.prior_means       # (K, L)
        prior_logvars = self.prior_logvars   # (K, L)
        prior_logits = self.prior_logits     # (K,)
        
        # Expand z to (N, 1, L) and prior to (1, K, L) where N = B*S
        z_ex = z.unsqueeze(1)
        means_ex = prior_means.unsqueeze(0)
        logvars_ex = prior_logvars.unsqueeze(0)
        
        # Calculate log component probabilities: (N, K)
        log_prob_components = -0.5 * (
            math.log(2 * math.pi) + 
            logvars_ex + 
            (z_ex - means_ex).pow(2) / logvars_ex.exp()
        ).sum(dim=2)
        
        # Add mixture weights (log_softmax of logits)
        log_mix_weights = F.log_softmax(prior_logits, dim=0).unsqueeze(0) # (1, K)
        
        # LogSumExp to get log p(z): (N,)
        log_p_z = torch.logsumexp(log_prob_components + log_mix_weights, dim=1)
        
        # KL Divergence: E_q [ log q(z|x) - log p(z) ]
        # kl = (log_q_z_x - log_p_z).mean() # TODO: Changed from below
        
        # return kl # TODO: Changed from below
        
        free_nats_per_dim = self.free_nats_per_dim
        free_nats_total = free_nats_per_dim * self.latent_dim

        kl_per = (log_q_z_x - log_p_z)          # (N,)
        kl_fb  = torch.clamp(kl_per, min=free_nats_total).mean()
        return kl_fb
    
    def standard_prior(self, logvar: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        kld = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        free_nats_per_dim = self.free_nats_per_dim
        # kld_fb = torch.clamp(kld- free_nats_per_dim, min=0).sum(dim=-1).mean()
        kld_fb = torch.clamp(kld, min=free_nats_per_dim).sum(dim=-1).mean()
        return kld_fb
    
    # Optional: helper for deterministic latent features (for HMM)
    def encode_to_latent(self, x, subject_ids):
        """
        Return deterministic latent representation (mu) for downstream HMM.
        """
        mu, _ = self.encode(x, subject_ids)
        return mu
    
    def regularization_loss(self, reg_loss, epoch) -> torch.Tensor:
        if self.max_beta == 0.0:
            return torch.tensor(0.0, device=self.device)

        total_epochs = int(getattr(self.global_config.trainer, "epochs", 0) or 0)
        last_epoch = max(0, total_epochs - 1)
        if total_epochs > 0 and epoch >= last_epoch:
            self.beta = self.max_beta
            return self.beta * reg_loss

        if self.no_beta_epochs and epoch <= self.no_beta_epochs:
            return torch.tensor(0.0, device=self.device)

        if self.beta_schedule == "cyclical" and self.beta_cyclical_period_epochs > 0:
            period = max(1, self.beta_cyclical_period_epochs)
            effective = max(0, epoch - (self.no_beta_epochs or 0))
            pos = effective % period
            progress = 0.5 * (1.0 - float(np.cos(np.pi * pos / period)))
            self.beta = self.min_beta + (self.max_beta - self.min_beta) * progress
        elif self.beta_warmup_epochs and epoch <= self.beta_warmup_epochs:
            warm_start = self.no_beta_epochs or 0
            warm_span = max(1, self.beta_warmup_epochs - warm_start)
            warm_progress = min(1.0, max(0.0, float(epoch - warm_start) / float(warm_span)))
            self.beta = self.min_beta + (self.max_beta - self.min_beta) * warm_progress
        elif self.beta_slowdown_epochs and self.beta_warmup_epochs and epoch > self.beta_warmup_epochs:
            excess_epochs = epoch - self.beta_warmup_epochs
            self.beta = max(
                self.min_beta,
                self.max_beta - (self.max_beta - self.min_beta) * excess_epochs / self.beta_slowdown_epochs,
            )
        else:
            self.beta = self.max_beta

        return self.beta * reg_loss
    
    
    ## USE GMM TO PREDICT CLUSTERS

    @torch.no_grad()
    def predict_gmm_labels(
        self,
        x: torch.Tensor,
        subject_ids: torch.Tensor,
        *,
        use_mu: bool = True,
        likelihood_type: str = "posterior",  # "posterior" | "log_posterior" | "joint" | "log_joint"
        initialize_if_needed: bool = False,
        max_points: int = 200_000,
    ):
        """
        Predict labels (GMM component indices) from the learned GMM prior via maximum likelihood.

        Args:
            x: (B, S, C, F)
            subject_ids: (B,) or (B, S)
            use_mu: If True, use encoder mean mu as latent z (deterministic).
                    If False, sample z via reparameterization (stochastic).
            likelihood_type:
                - "posterior":     p(k* | z)          (recommended)
                - "log_posterior": log p(k* | z)
                - "joint":         p(z, k*) = pi_k* N(z|...)
                - "log_joint":     log p(z, k*)
            initialize_if_needed: If prior is "warm_gmm" and not initialized, optionally run KMeans init.
            max_points: Passed to KMeans init if used.

        Returns:
            y: (B, S) LongTensor of predicted component indices
            likelihood: (B, S) Tensor of chosen likelihood quantity (see likelihood_type)
        """
        # Marginal per-step GMM assignment uses emission params shared with HMM-GMM priors.
        if self.prior not in ("gmm", "warm_gmm", "hmm_gmm", "warm_hmm_gmm"):
            raise ValueError(
                f"predict_gmm_labels requires a GMM or HMM-GMM emission prior, got prior='{self.prior}'."
            )

        # If warm_* and not initialized, optionally initialize from current encoder means.
        if self.prior in ("warm_gmm", "warm_hmm_gmm") and (
            not getattr(self, "gmm_warmup_initialized", False)
        ):
            if not initialize_if_needed:
                raise RuntimeError(
                    "GMM prior not initialized (warm_gmm). "
                    "Run training past gmm_warmup_epochs or call with initialize_if_needed=True."
                )
            means, logvars, logits = self.__get_kmeans_centroids(max_points=max_points)
            self.prior_means.copy_(means)
            self.prior_logvars.copy_(logvars)
            self.prior_logits.copy_(logits)
            self.gmm_warmup_initialized = True

        self.eval()

        B, S, _, _ = x.shape
        x = x.to(self.device)

        # Encode to latent
        mu, logvar = self.encode(x, subject_ids=subject_ids)
        z = mu if use_mu else self.reparameterize(mu, logvar)  # (B, S, L)

        # Flatten to (N, L) where N = B*S
        z_flat = z.reshape(-1, self.latent_dim)

        # Compute log p(z | k) for each component k
        # Shapes:
        #   z_ex:        (N, 1, L)
        #   means_ex:    (1, K, L)
        #   logvars_ex:  (1, K, L)
        z_ex = z_flat.unsqueeze(1)
        means_ex = self.prior_means.unsqueeze(0)
        logvars_ex = self.prior_logvars.unsqueeze(0)

        log_prob_components = -0.5 * (
            math.log(2 * math.pi) +
            logvars_ex +
            (z_ex - means_ex).pow(2) / logvars_ex.exp()
        ).sum(dim=2)  # (N, K)

        # Add log mixture weights
        log_mix = F.log_softmax(self.prior_logits, dim=0).unsqueeze(0)  # (1, K)
        log_joint = log_prob_components + log_mix  # (N, K)  == log p(z, k)

        # ML label: argmax_k log p(z, k)
        y_flat = torch.argmax(log_joint, dim=1)  # (N,)

        # Gather chosen component scores
        log_joint_chosen = log_joint.gather(1, y_flat.unsqueeze(1)).squeeze(1)  # (N,)

        # Posterior of chosen label: p(k* | z) = exp(log p(z,k*) - logsumexp_k log p(z,k))
        log_norm = torch.logsumexp(log_joint, dim=1)  # (N,)
        log_post_chosen = log_joint_chosen - log_norm  # (N,)

        if likelihood_type == "posterior":
            lik_flat = log_post_chosen.exp()
        elif likelihood_type == "log_posterior":
            lik_flat = log_post_chosen
        elif likelihood_type == "joint":
            lik_flat = log_joint_chosen.exp()
        elif likelihood_type == "log_joint":
            lik_flat = log_joint_chosen
        else:
            raise ValueError(
                "likelihood_type must be one of: "
                "'posterior', 'log_posterior', 'joint', 'log_joint'"
            )
        
        # Reshape back to (B, S)
        y = y_flat.view(B, S).long()
        likelihood = lik_flat.view(B, S)
        # expected likelihood
        likelihood = likelihood.mean()
        
        # Latent prior log-likelihood: log p(z) = logsumexp_k log p(z,k)
        log_pz_flat = torch.logsumexp(log_joint, dim=1)                # (N,)
        log_pz = log_pz_flat.view(B, S)
        log_pz = log_pz.mean()

        return y, log_pz, mu

    @torch.no_grad()
    def predict_hmm_labels(
        self,
        x: torch.Tensor,
        subject_ids: torch.Tensor,
        *,
        use_mu: bool = True,
    ):
        """Viterbi-decode HMM-GMM state path on the latent sequence.

        Args:
            x: (B, T, C, F)
            subject_ids: (B,) or (B, T)
            use_mu: deterministic mu vs sampled z.

        Returns:
            y:      (B, T) LongTensor of state indices
            log_pz: scalar mean log p(z_{1:T}) under the HMM prior
            mu:     (B, T, L) deterministic latent mean
        """
        if self.prior not in ("hmm_gmm", "warm_hmm_gmm"):
            raise ValueError(
                f"predict_hmm_labels requires an HMM-GMM prior, got prior='{self.prior}'."
            )

        self.eval()
        x = x.to(self.device)
        mu, logvar = self.encode(x, subject_ids=subject_ids)
        z = mu if use_mu else self.reparameterize(mu, logvar)

        log_emit = gaussian_log_prob_diag(z, self.prior_means, self.prior_logvars)
        log_pi = F.log_softmax(self.prior_logits, dim=0)
        log_A = F.log_softmax(self.prior_transition_logits, dim=-1)

        y = viterbi_decode(log_emit, log_pi, log_A)
        log_pz = forward_log_marginal(log_emit, log_pi, log_A).mean()
        return y, log_pz, mu