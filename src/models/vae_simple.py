
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
        
        # Latent dimensions
        self.latent_dim = params.get("latent_dim", None)
        self.enc_hidden_dims = params.get("enc_hidden_dims", None) # MLP hidden layers
        self.dec_hidden_dims = params.get("dec_hidden_dims", None) # MLP hidden layers
        self.emb_dim = params.get("emb_dim", None)
        
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
        
        # Set torch random seed for reproducibility
        seed = self.global_config.seed
        torch.manual_seed(seed)

    def __initialize_weights(self):
        
        # ----- Subject Embedding -----
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
        prev = self.C * self.F + self.emb_dim
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
        mlp.append(nn.Linear(prev, self.C * self.F))
        self.decoder_mlp = nn.Sequential(*mlp)
        
        # GMM prior parameters (unchanged)
        if self.prior in ("gmm","warm_gmm"):
            self.prior_logits = nn.Parameter(torch.zeros(self.num_states, device=self.device))
            self.prior_means = nn.Parameter(torch.randn(self.num_states, self.latent_dim))
            self.prior_logvars = nn.Parameter(torch.zeros(self.num_states, self.latent_dim, device=self.device))

        self.to(self.device)

    def reset(self):
        self.__initialize_weights()

    # ---------- Core subroutines ----------
    
    def get_subject_embedding(self, subject_ids):
        """
        subject_ids: (B,) or (B, S)
        returns: emb with shape (B*S, emb_dim)
        """
        B_S = subject_ids.numel()
        if subject_ids.dim() == 1:
            subject_ids_expanded = subject_ids.repeat_interleave(self.S)
        elif subject_ids.dim() == 2:
             subject_ids_expanded = subject_ids.view(-1)
        else:
            subject_ids_expanded = subject_ids
        emb = self.subject_emb(subject_ids_expanded)
        return emb
    
    def encode(self, x, subject_ids):
        """
        x: (B, S, C, F)
        subject_ids: (B,) or (B, S)
        returns: mu, logvar both (B, S, L)
        """
        B, S, C, F = x.shape
        
        # Get subject embeddings
        emb = self.get_subject_embedding(subject_ids)
        
        # Flatten batch and sequence for independent processing
        x_flat = x.view(B * S, -1) # (B*S, C*F)
        
        # Add subject embedding
        h_flat = torch.cat([x_flat, emb], dim=-1)
        
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
        emb = self.get_subject_embedding(subject_ids)
        
        z_flat = z.view(B * S, L)
        h_in = torch.cat([z_flat, emb], dim=-1)
        
        # MLP
        h_flat = self.decoder_mlp(h_in)
        
        # Reshape back
        x_recon = h_flat.view(B, S, self.C, self.F)
        
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
            print(f"Initialized GMM prior with KMeans at epoch {epoch}")
            print(f"  Means: {means}")
            print(f"  Logvars: {logvars}")
            print(f"  Logits: {logits}")
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
        kl = (log_q_z_x - log_p_z).mean()
        
        return kl

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
        mu, _ = self.encode(x, subject_ids)
        return mu
    
    def regularization_loss(self, epoch) -> torch.Tensor:
        if self.no_beta_epochs and epoch <= self.no_beta_epochs:
            return torch.tensor(0.0, device=self.device)
        self.beta = min(self.max_beta, self.min_beta + (self.max_beta - self.min_beta) * epoch / (self.beta_warmup_epochs-self.no_beta_epochs)) if self.beta_warmup_epochs else self.max_beta
        reg = self.beta * self.reg_loss
        return reg