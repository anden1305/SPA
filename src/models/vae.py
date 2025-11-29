
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from src.config.config import GlobalConfig
from src.data.data_loader_collection import DataLoaderCollection

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
        self.seed = self.global_config.seed
        torch.manual_seed(int(self.seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(self.seed))
        params = self.global_config.model.params
        self.n_subjects = self.data_loader.get_num_subjects()
        self.input_dim = self.data_loader.get_feature_dim()
        self.latent_dim = params.get("latent_dim", None)
        self.hidden_dims = params.get("hidden_dims", None)
        self.emb_dim = params.get("emb_dim", None)
        # convolutional parameters
        self.use_conv = params.get("use_conv", True)
        self.seq_len = params.get("sequence_length", 512)
        self.n_channels = self.data_loader.get_channels()
        # beta
        self.min_beta = params.get("min_beta", 0.1)
        self.max_beta = params.get("max_beta", 1.0)
        self.beta_warmup = params.get("beta_warmup", 0)
        self.beta = self.min_beta
        self.current_step = 0
        # subjects
        self.subject_conditioning = params.get("subject_conditioning", True)
        if not self.subject_conditioning:
            self.emb_dim = 0
        # Prior
        self.prior = params.get("prior", "standard_normal")
        
        assert self.prior in ["standard_normal", "gmm", None], "Prior must be 'standard_normal', 'gmm', or 'null'"
        if self.prior == "gmm":
            self.n_components = self.data_loader.get_num_states()
    
    # def __initialize_weights(self):
    #     # 1) Subject ID embedding
    #     if self.subject_conditioning:
    #         self.subject_emb = nn.Embedding(num_embeddings=self.n_subjects, embedding_dim=self.emb_dim)
    #     # 2) Build encoder MLP
    #     enc_layers = []
    #     prev_dim = self.input_dim + self.emb_dim  # x concatenated with subject embedding
    #     for h in self.hidden_dims:
    #         enc_layers.append(nn.Linear(prev_dim, h))
    #         enc_layers.append(nn.ReLU())
    #         prev_dim = h
    #     self.encoder_mlp = nn.Sequential(*enc_layers)
    #     # 3) Latent heads: mean and log-variance
    #     self.fc_mu = nn.Linear(prev_dim, self.latent_dim)
    #     self.fc_logvar = nn.Linear(prev_dim, self.latent_dim)
    #     # 4) Build decoder MLP (mirror encoder)
    #     dec_layers = []
    #     prev_dim = self.latent_dim + self.emb_dim  # z concatenated with mouse embedding
    #     for h in reversed(self.hidden_dims):
    #         dec_layers.append(nn.Linear(prev_dim, h))
    #         dec_layers.append(nn.ReLU())
    #         prev_dim = h
    #     dec_layers.append(nn.Linear(prev_dim, self.input_dim))  # final reconstruction layer
    #     self.decoder_mlp = nn.Sequential(*dec_layers)
    #     # 5) GMM prior parameters
    #     if self.prior == "gmm":
    #         # mixture weights (unnormalized logits)
    #         self.prior_logits = nn.Parameter(
    #             torch.zeros(self.n_components)  # will be softmaxed
    #         )
    #         # component means (K, D)
    #         self.prior_means = nn.Parameter(
    #             torch.randn(self.n_components, self.latent_dim) * 0.1
    #         )
    #         # component log-variances (K, D)
    #         self.prior_logvars = nn.Parameter(
    #             torch.zeros(self.n_components, self.latent_dim)
    #         )
    #     # 6) To device
    #     self.to(self.device)
    
    def __initialize_weights(self):
        # 1) Subject ID embedding (same as before)
        if self.subject_conditioning:
            self.subject_emb = nn.Embedding(
                num_embeddings=self.n_subjects,
                embedding_dim=self.emb_dim
            )

        if not self.use_conv:
            # -------- ORIGINAL MLP-ONLY VERSION --------
            enc_layers = []
            prev_dim = self.input_dim + self.emb_dim  # x concatenated with subject embedding
            for h in self.hidden_dims:
                enc_layers.append(nn.Linear(prev_dim, h))
                enc_layers.append(nn.ReLU())
                prev_dim = h
            self.encoder_mlp = nn.Sequential(*enc_layers)

            self.fc_mu = nn.Linear(prev_dim, self.latent_dim)
            self.fc_logvar = nn.Linear(prev_dim, self.latent_dim)

            # Decoder MLP (mirror encoder)
            dec_layers = []
            prev_dim = self.latent_dim + self.emb_dim
            for h in reversed(self.hidden_dims):
                dec_layers.append(nn.Linear(prev_dim, h))
                dec_layers.append(nn.ReLU())
                prev_dim = h
            dec_layers.append(nn.Linear(prev_dim, self.input_dim))
            self.decoder_mlp = nn.Sequential(*dec_layers)
        else:
            # -------- NEW: CNN + MLP HYBRID VERSION --------
            # a) Conv encoder working on (B, C, T)
            conv_channels = self.global_config.model.params.get(
                "conv_channels", [32, 64]
            )
            self.conv_encoder = nn.Sequential(
                nn.Conv1d(self.n_channels, conv_channels[0],
                        kernel_size=5, stride=2, padding=2),
                nn.ReLU(),
                nn.Conv1d(conv_channels[0], conv_channels[1],
                        kernel_size=5, stride=2, padding=2),
                nn.ReLU()
            )
            self.conv_channels = conv_channels

            # Infer conv encoder output size
            with torch.no_grad():
                dummy = torch.zeros(1, self.n_channels, self.seq_len)
                out = self.conv_encoder(dummy)
                self.conv_out_channels = out.size(1)
                self.conv_out_len = out.size(2)
                self.encoder_flat_dim = self.conv_out_channels * self.conv_out_len

            # b) Encoder MLP on flattened conv features
            enc_layers = []
            prev_dim = self.encoder_flat_dim + self.emb_dim
            for h in self.hidden_dims:
                enc_layers.append(nn.Linear(prev_dim, h))
                enc_layers.append(nn.ReLU())
                prev_dim = h
            self.encoder_mlp = nn.Sequential(*enc_layers)

            self.fc_mu = nn.Linear(prev_dim, self.latent_dim)
            self.fc_logvar = nn.Linear(prev_dim, self.latent_dim)

            # c) Decoder: latent -> flattened conv features -> ConvTranspose1d
            # First, map (z [+ emb]) to flattened conv feature space
            self.decoder_fc = nn.Sequential(
                nn.Linear(self.latent_dim + self.emb_dim, prev_dim),
                nn.ReLU(),
                nn.Linear(prev_dim, self.encoder_flat_dim),
                nn.ReLU()
            )

            # Conv decoder: mirror conv_encoder
            self.conv_decoder = nn.Sequential(
                nn.ConvTranspose1d(
                    self.conv_out_channels, conv_channels[0],
                    kernel_size=5, stride=2, padding=2, output_padding=1
                ),
                nn.ReLU(),
                nn.ConvTranspose1d(
                    conv_channels[0], self.n_channels,
                    kernel_size=5, stride=2, padding=2, output_padding=1
                )
                # no activation -> we let MSE handle scale
            )

        # 5) GMM prior parameters (unchanged)
        if self.prior == "gmm":
            self.prior_logits = nn.Parameter(torch.zeros(self.n_components))
            self.prior_means = nn.Parameter(
                torch.randn(self.n_components, self.latent_dim) * 0.1
            )
            self.prior_logvars = nn.Parameter(
                torch.zeros(self.n_components, self.latent_dim)
            )

        self.to(self.device)

        
        
    def reset(self):
        self.__initialize_weights()

    # ---------- Core subroutines ----------

    # def encode(self, x, subject_ids):
    #     """
    #     x:          (B, input_dim)
    #     subject_ids:  (B,) long tensor
    #     returns: mu, logvar  both (B, latent_dim)
    #     """
    #     if self.subject_conditioning:
    #         emb = self.subject_emb(subject_ids)   
    #         h_in = torch.cat([x, emb], dim=-1)        # (B, input_dim + emb_dim)
    #     else:
    #         h_in = x
    #     h = self.encoder_mlp(h_in)               # (B, last_hidden)
    #     mu = self.fc_mu(h)
    #     logvar = self.fc_logvar(h)
    #     return mu, logvar
    
    def encode(self, x, subject_ids):
        """
        If use_conv:
            x: (B, T, C) raw EEG/EMG
        else:
            x: (B, input_dim)
        """
        if self.use_conv:
            h_feat = self.conv_encoder(x)          # (B, F, T')
            h_feat = h_feat.flatten(1)             # (B, F * T')
        else:
            h_feat = x                             # (B, input_dim)
        
        if self.subject_conditioning:
            emb = self.subject_emb(subject_ids)    # (B, emb_dim)
            h_in = torch.cat([h_feat, emb], dim=-1)
        else:
            h_in = h_feat

        h = self.encoder_mlp(h_in)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    
    
    def reparameterize(self, mu, logvar):
        """
        Standard VAE reparameterization.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    # def decode(self, z, subject_ids):
    #     """
    #     z:          (B, latent_dim)
    #     subject_ids:  (B,)
    #     returns: x_recon with shape (B, input_dim)
    #     """
    #     if self.subject_conditioning:
    #         emb = self.subject_emb(subject_ids)          # (B, emb_dim)
    #         h_in = torch.cat([z, emb], dim=-1)       # (B, latent_dim + emb_dim)
    #     else:
    #         h_in = z
    #     x_recon = self.decoder_mlp(h_in)         # (B, input_dim)
    #     return x_recon
    
    def decode(self, z, subject_ids):
        """
        If use_conv:
            returns x_recon: (B, T, C) raw
        else:
            returns x_recon: (B, input_dim)
        """
        if self.subject_conditioning:
            emb = self.subject_emb(subject_ids)   # (B, emb_dim)
            h_in = torch.cat([z, emb], dim=-1)
        else:
            h_in = z

        if not self.use_conv:
            # original MLP-only decoder
            x_recon = self.decoder_mlp(h_in)       # (B, input_dim)
            return x_recon

        # CNN decoder path
        h = self.decoder_fc(h_in)                  # (B, F * T')
        B = h.size(0)
        h = h.view(B, self.conv_out_channels, self.conv_out_len)  # (B, F, T')

        x_recon = self.conv_decoder(h)             # (B, C, T_decoded)
        # # If T_decoded > seq_len, crop; if smaller, you can pad or just accept if equal.
        # if x_recon.size(-1) > self.seq_len:
        #     x_recon = x_recon[..., :self.seq_len]
        # elif x_recon.size(-1) < self.seq_len:
        #     # simple padding with zeros (or reflect, etc.)
        #     pad_len = self.seq_len - x_recon.size(-1)
        #     x_recon = F.pad(x_recon, (0, pad_len))

        # back to (B, T, C) to match the input
        return x_recon

    
    
    
    def forward(self, x, subject_ids):
        """
        Full VAE pass:
        - encode → mu, logvar
        - reparam → z
        - decode(z, subject_ids) → x_recon

        Returns:
            x_recon: (B, input_dim)
            mu:      (B, latent_dim)
            logvar:  (B, latent_dim)
            z:       (B, latent_dim)
        """
        mu, logvar = self.encode(x, subject_ids)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z, subject_ids)
        return x_recon, mu, logvar, z
    
    
    # def calculate_loss(self, x, x_recon, mu, logvar):
    #     """
    #     Compute VAE loss with optional GMM prior.
    #     """
    #     # 1. Reconstruction Loss (MSE)
    #     recon_loss = F.mse_loss(x_recon, x, reduction='none').sum(dim=-1).mean()

    #     # 2. KL Divergence Loss
    #     if self.prior == "gmm":
    #         self.kld_loss = self._gmm_kl(mu, logvar)
    #     elif self.prior == "standard_normal":
    #         # fallback: standard Normal prior
    #         self.kld_loss = -0.5 * torch.sum(
    #             1 + logvar - mu.pow(2) - logvar.exp(), dim=-1
    #         ).mean()
    #     elif self.prior is None:
    #         pass
    #     else:
    #         raise ValueError(f"Unknown prior type: {self.prior}")

    #     return recon_loss
    
    def calculate_loss(self, x, x_recon, mu, logvar):
        """
        Compute VAE loss with optional GMM prior.
        """
        # 1. Reconstruction Loss (MSE)
        if self.use_conv:
            # CONV
            recon_loss = F.mse_loss(
                x_recon, x, reduction='none'
            ).sum(dim=(1, 2)).mean()
        else:
            # NON-CONV
            recon_loss = F.mse_loss(
                x_recon, x, reduction='none'
            ).sum(dim=-1).mean()

        # 2. KL Divergence Loss
        if self.prior == "gmm":
            self.kld_loss = self._gmm_kl(mu, logvar)
        elif self.prior == "standard_normal":
            # fallback: standard Normal prior
            self.kld_loss = -0.5 * torch.sum(
                1 + logvar - mu.pow(2) - logvar.exp(), dim=-1
            ).mean()
        elif self.prior is None:
            pass
        else:
            raise ValueError(f"Unknown prior type: {self.prior}")

        return recon_loss


    def _log_normal_diag(self, z, mu, logvar):
        """
        z, mu, logvar: (..., D)
        Returns log N(z | mu, diag(exp(logvar)))  => (...,)
        """
        D = z.size(-1)
        const = D * math.log(2 * math.pi)
        return -0.5 * (
            const
            + logvar.sum(dim=-1)
            + ((z - mu) ** 2 / logvar.exp()).sum(dim=-1)
        )

    
    def _log_gmm_prior(self, z):
        """
        z: (..., D)
        prior_means: (K, D)
        prior_logvars: (K, D)
        prior_logits: (K,)
        Returns log p(z) where p is GMM => (...,)
        """
        # Batch shape (could be (B,), (B, T), etc.), and latent dim
        batch_shape = z.shape[:-1]      # e.g. (B, T)
        D = z.shape[-1]                 # latent_dim
        K = self.n_components

        # Flatten batch dimensions so z_flat: (N, D)
        N = z.numel() // D
        z_flat = z.reshape(N, D)        # (N, D)

        # Prior params: (K, D)
        mu = self.prior_means.view(1, K, D)        # (1, K, D)
        logvar = self.prior_logvars.view(1, K, D)  # (1, K, D)

        # z -> (N, 1, D)
        z_expand = z_flat.unsqueeze(1)             # (N, 1, D)

        const = D * math.log(2 * math.pi)

        # log N(z | mu_k, Sigma_k) for each component -> (N, K)
        log_prob_per_comp = -0.5 * (
            const
            + logvar.sum(dim=-1)                                  # (1, K)
            + ((z_expand - mu) ** 2 / logvar.exp()).sum(dim=-1)   # (N, K)
        )

        # mixture weights
        log_pi = F.log_softmax(self.prior_logits, dim=0)  # (K,)
        log_pi = log_pi.unsqueeze(0)                      # (1, K)

        # log p(z) = logsum_k pi_k N(z | mu_k, Sigma_k)
        log_pz_flat = torch.logsumexp(log_pi + log_prob_per_comp, dim=1)  # (N,)

        # Reshape back to original batch shape (...,)
        log_pz = log_pz_flat.view(batch_shape)
        return log_pz


    def _gmm_kl(self, mu, logvar):
        """
        Monte Carlo KL(q(z|x) || p(z)) with GMM prior.
        mu, logvar : (..., D)
        """
        z = self.reparameterize(mu, logvar)            # (..., D)

        log_qzx = self._log_normal_diag(z, mu, logvar) # (...,)
        log_pz = self._log_gmm_prior(z)                # (...,)

        kl = (log_qzx - log_pz).mean()                 # scalar
        return kl
    
    def encode_to_latent(self, x, subject_ids):
        """
        Return deterministic latent representation (mu) for downstream HMM.
        """
        mu, _ = self.encode(x, subject_ids)
        return mu
    
    def regularization_loss(self) -> torch.Tensor:
        if self.prior is None:
            return torch.tensor(0.0, device=self.device)
        reg_loss = self.kld_loss * self.beta
        self.update_beta()
        return reg_loss
    
    def update_beta(self):
        if self.beta_warmup > 0:
            beta_increment = (self.max_beta - self.min_beta) / self.beta_warmup
            new_beta = min(self.min_beta + beta_increment * self.current_step, self.max_beta)
            self.beta = new_beta
            self.current_step += 1