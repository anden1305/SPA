
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.input_dim = self.data_loader.get_feature_dim()
        self.latent_dim = params.get("latent_dim", None)
        self.hidden_dims = params.get("hidden_dims", None)
        self.emb_dim = params.get("emb_dim", None)
        
    def __initialize_weights(self):
        # 1) Subject ID embedding
        self.subject_emb = nn.Embedding(num_embeddings=self.n_subjects, embedding_dim=self.emb_dim)
        # 2) Build encoder MLP
        enc_layers = []
        prev_dim = self.input_dim + self.emb_dim  # x concatenated with subject embedding
        for h in self.hidden_dims:
            enc_layers.append(nn.Linear(prev_dim, h))
            enc_layers.append(nn.ReLU())
            prev_dim = h
        self.encoder_mlp = nn.Sequential(*enc_layers)
        
        # 3) Latent heads: mean and log-variance
        self.fc_mu = nn.Linear(prev_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, self.latent_dim)

        # 4) Build decoder MLP (mirror encoder)
        dec_layers = []
        prev_dim = self.latent_dim + self.emb_dim  # z concatenated with mouse embedding
        for h in reversed(self.hidden_dims):
            dec_layers.append(nn.Linear(prev_dim, h))
            dec_layers.append(nn.ReLU())
            prev_dim = h
        dec_layers.append(nn.Linear(prev_dim, self.input_dim))  # final reconstruction layer
        self.decoder_mlp = nn.Sequential(*dec_layers)
        self.to(self.device)
        
    def reset(self):
        self.__initialize_weights()

    # ---------- Core subroutines ----------

    def encode(self, x, subject_ids):
        """
        x:          (B, input_dim)
        subject_ids:  (B,) long tensor
        returns: mu, logvar  both (B, latent_dim)
        """
        emb = self.subject_emb(subject_ids)   
        h_in = torch.cat([x, emb], dim=-1)        # (B, input_dim + emb_dim)
        h = self.encoder_mlp(h_in)               # (B, last_hidden)
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

    def decode(self, z, subject_ids):
        """
        z:          (B, latent_dim)
        subject_ids:  (B,)
        returns: x_recon with shape (B, input_dim)
        """
        emb = self.subject_emb(subject_ids)          # (B, emb_dim)
        h_in = torch.cat([z, emb], dim=-1)       # (B, latent_dim + emb_dim)
        x_recon = self.decoder_mlp(h_in)         # (B, input_dim)
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
    
    def calculate_loss(self, x, x_recon, mu, logvar):
        """
        Compute VAE loss: reconstruction + KL divergence.
        """
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(x_recon, x, reduction='mean')
        # KL divergence loss
        kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        total_loss = recon_loss + kld_loss
        return total_loss

    # Optional: helper for deterministic latent features (for HMM)
    def encode_to_latent(self, x, subject_ids):
        """
        Return deterministic latent representation (mu) for downstream HMM.
        """
        mu, logvar = self.encode(x, subject_ids)
        return mu

    def regularization_loss(self) -> torch.Tensor:
        return torch.tensor(0.0, device=self.device)