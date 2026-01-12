
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans

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
        params = self.global_config.model.params
        self.n_subjects = self.data_loader.get_num_subjects()
        
        # Dimensions (Hardcoded defaults as requested, but overridable via params)
        C, F, S, num_states = self.data_loader.get_vae_dims()
        self.S = S
        self.C = C
        self.F = F
        self.num_states = num_states
        
        self.latent_dim = params.get("latent_dim", None)
        self.enc_hidden_dims = params.get("enc_hidden_dims", None) # MLP hidden layers
        self.dec_hidden_dims = params.get("dec_hidden_dims", None) # MLP hidden layers
        self.emb_dim = params.get("emb_dim", None)
        
        # # Convolutional config
        self.conv_channels = params.get("conv_channels", None)
        self.kernel_sizes = params.get("kernel_sizes", None)
        self.strides      = params.get("strides", None)
        self.paddings     = params.get("paddings", None)
        
        # Beta
        self.min_beta = params.get("min_beta", None)
        self.max_beta = params.get("max_beta", None)
        self.beta = self.min_beta
        self.beta_warmup_epochs = params.get("beta_warmup_epochs", None)
        self.no_beta_epochs = params.get("no_beta_epochs", None)
        
        # Prior
        self.prior = params.get("prior", "warm_gmm") # "standard", "gmm" or "warm_gmm"
        self.gmm_warmup_epochs = params.get("gmm_warmup_epochs", 50) # only for "warm_gmm"
        self.gmm_warmup_initialized = False
        
        # RMS skip connection
        self.use_rms = params.get("use_rms", False)
            
        # Set torch random seed for reproducibility
        seed = self.global_config.seed
        torch.manual_seed(seed)

    def __initialize_weights(self):
        self.subject_emb = nn.Embedding(self.n_subjects, self.emb_dim)

        # ----- Encoder CNN -----
        enc_layers = []
        in_ch = self.C
        L = self.F
        self._lens = [L]  # store lengths through encoder (for decoder)
        for i, (out_ch, k, s, p) in enumerate(zip(self.conv_channels, self.kernel_sizes, self.strides, self.paddings)):
            enc_layers.append(nn.Conv1d(in_ch, out_ch, kernel_size=k, stride=s, padding=p))
            enc_layers.append(nn.ReLU())
            in_ch = out_ch
            L = self._conv1d_out_len(L, k, s, p)
            self._lens.append(L)

        self.encoder_cnn = nn.Sequential(*enc_layers)

        # Flat dim via dummy pass (still fine)
        with torch.no_grad():
            dummy = torch.zeros(1, self.C, self.F)
            out = self.encoder_cnn(dummy)
            self.flat_dim = out.view(1, -1).shape[1]
            self.conv_out_shape = out.shape[1:]  # (ch, len)

        # ----- Encoder MLP -----
        mlp = []
        prev = self.flat_dim + (3 if self.use_rms else 0) + self.emb_dim # +3 for RMS features if use_rms is True
        for h in self.enc_hidden_dims:
            mlp += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        self.encoder_mlp = nn.Sequential(*mlp)

        self.fc_mu = nn.Linear(prev, self.latent_dim)
        self.fc_logvar = nn.Linear(prev, self.latent_dim)
        
        # ----- Decoder MLP -----
        mlp = []
        prev = self.latent_dim + self.emb_dim
        for h in self.dec_hidden_dims:
            mlp += [nn.Linear(prev, h), nn.ReLU()]
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

            out_ch = self.C if i == len(rev_channels) - 1 else rev_channels[i + 1]
            dec_layers.append(nn.ConvTranspose1d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, output_padding=out_pad))
            if not (i == len(rev_channels) - 1):
                dec_layers.append(nn.ReLU())
            # update for next layer
            L_in = target_L
            in_ch = out_ch
        
        self.decoder_cnn = nn.Sequential(*dec_layers)
        
        # GMM prior parameters (unchanged)
        if self.prior == "gmm":
            self.prior_logits = nn.Parameter(torch.zeros(self.num_states, device=self.device))
            self.prior_means = nn.Parameter(torch.randn(self.num_states, self.latent_dim))
            self.prior_logvars = nn.Parameter(torch.zeros(self.num_states, self.latent_dim, device=self.device))
        
        if self.prior == "warm_gmm":
            self.prior_logits = nn.Parameter(torch.zeros(self.num_states))
            self.prior_means = nn.Parameter(torch.randn(self.num_states, self.latent_dim) * 0.1)
            self.prior_logvars = nn.Parameter(torch.zeros(self.num_states, self.latent_dim))

        self.to(self.device)
        
        
    def _conv1d_out_len(self, L, k, s, p, d=1):
        # PyTorch Conv1d output length
        return (L + 2*p - d*(k-1) - 1) // s + 1
    

    def reset(self):
        self.__initialize_weights()

    # ---------- Core subroutines ----------

    def encode(self, x, subject_ids):
        """
        x: (B, S, C, F)
        subject_ids: (B,) or (B, S)
        returns: mu, logvar both (B, S, L)
        """
        B, S, C, F = x.shape
        
        if subject_ids.dim() == 1:
            subject_ids_expanded = subject_ids.repeat_interleave(S)
        elif subject_ids.dim() == 2:
             subject_ids_expanded = subject_ids.view(-1)
        else:
            subject_ids_expanded = subject_ids
        
        emb = self.subject_emb(subject_ids_expanded)
        
        # Flatten batch and sequence for independent processing
        x_flat = x.view(B * S, C, F)
        
        # CNN
        h_conv = self.encoder_cnn(x_flat) # (B*S, last_ch, last_len)
        h_flat = h_conv.view(B * S, -1)
        
        # Add subject embedding
        h_flat = torch.cat([h_flat, emb], dim=-1)
        
        if self.use_rms:
            # Calculate RMS for each channel
            rms_flat = torch.sqrt(torch.mean(x_flat ** 2, dim=-1, keepdim=True)).view(B * S, 3)
            # Add RMS as additional feature
            h_flat = torch.cat([h_flat, rms_flat], dim=-1)  # now shape (B*S, C, F+1)
        
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
        z_flat = z.view(B * S, L)
        
        if subject_ids.dim() == 1:
            subject_ids_expanded = subject_ids.repeat_interleave(S)
        elif subject_ids.dim() == 2:
             subject_ids_expanded = subject_ids.view(-1)
        else:
            subject_ids_expanded = subject_ids
        
        emb = self.subject_emb(subject_ids_expanded)
        
        h_in = torch.cat([z_flat, emb], dim=-1)
        
        # MLP
        h_flat = self.decoder_mlp(h_in)
        
        # Reshape for CNN
        h_conv_in = h_flat.view(B * S, *self.conv_out_shape)
        
        # CNN Transpose
        x_recon_flat = self.decoder_cnn(h_conv_in)
        
        # Reshape back
        x_recon = x_recon_flat.view(B, S, self.C, self.F)
        
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
        recon_loss = F.mse_loss(x_recon, x, reduction='mean')
        # recon_loss = F.mse_loss(
        #     x_recon, x, reduction='none'
        # ).sum(dim=(-1, -2)).mean()
        # Regularization loss
        if self.prior == "gmm":
            self.reg_loss = self.gmm_prior(logvar, mu, z)
        elif self.prior == "standard":
            self.reg_loss = self.standard_prior(logvar, mu)
        elif self.prior == "warm_gmm":
            self.reg_loss = self.warm_gmm_prior(logvar, mu, z, epoch)
        return recon_loss
    
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
            n_clusters=self.num_states,
            n_init="auto",
            random_state=self.global_config.seed,
        )
        labels = km.fit_predict(Z_np)
        centroids_np = km.cluster_centers_

        # Back to torch on device
        centroids = torch.tensor(centroids_np, device=self.device, dtype=Z.dtype)

        # Mixture weights -> logits
        counts = np.bincount(labels, minlength=self.num_states).astype(np.float64)
        probs = counts / (counts.sum() + 1e-12)
        logits = torch.tensor(np.log(probs + 1e-12), device=self.device, dtype=Z.dtype)
        
        # Per-cluster diagonal variances -> logvars
        logvars = torch.zeros(self.num_states, Z.size(-1), device=self.device, dtype=Z.dtype)
        labels_t = torch.tensor(labels, device=self.device)
        
        var_floor = 1e-2

        for k in range(self.num_states):
            mask = labels_t == k
            if mask.any():
                diff = Z[mask] - centroids[k]
                var = diff.pow(2).mean(dim=0).clamp_min(var_floor)
                logvars[k] = var.log()
            else:
                logvars[k] = torch.zeros(Z.size(-1), device=self.device, dtype=Z.dtype)
                
        # overwrite logvars with reasonable value
        logvars = torch.full((self.num_states, Z.size(-1)), math.log(0.5**2), device=self.device, dtype=Z.dtype)

        self.train(was_training)
        return centroids, logvars, logits
    
    def gmm_prior(self, logvar: torch.Tensor, mu: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Monte Carlo KL(q(z|x) || p(z)) with GMM prior.
        mu, logvar : (..., D)
        """
        
        log_qzx = self.__log_normal_diag(z, mu, logvar) # (...,)
        log_pz = self.__log_gmm_prior(z)                # (...,)

        kl = (log_qzx - log_pz)
        free_nats_total = 0.5 * self.latent_dim
        kl_fb = torch.clamp(kl, min=free_nats_total).mean()
        return kl_fb * 0.1
    
    def __log_normal_diag(self, z, mu, logvar):
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
    
    def __log_gmm_prior(self, z):
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
        K = self.num_states

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

    def standard_prior(self, logvar: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        kld = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        free_nats_per_dim = 0.5
        kld_fb = torch.clamp(kld - free_nats_per_dim, min=0.0).sum(dim=-1).mean()
        return kld_fb
    
    # Optional: helper for deterministic latent features (for HMM)
    def encode_to_latent(self, x, subject_ids):
        """
        Return deterministic latent representation (mu) for downstream HMM.
        """
        mu, logvar = self.encode(x, subject_ids)
        return mu
    
    def regularization_loss(self, epoch) -> torch.Tensor:
        if self.no_beta_epochs and epoch <= self.no_beta_epochs:
            return torch.tensor(0.0, device=self.device)
        self.beta = min(self.max_beta, self.min_beta + (self.max_beta - self.min_beta) * epoch / (self.beta_warmup_epochs-self.no_beta_epochs)) if self.beta_warmup_epochs else self.max_beta
        reg = self.beta * self.reg_loss
        return reg
        # return self.kld_loss * beta