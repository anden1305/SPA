
from .base_model import BaseModel
import torch

from src.config.config import GlobalConfig
from src.data.data_loader_collection import DataLoaderCollection
from src.models.marhmm import MARHMM
from src.models.vae import ConditionalVAE

class CVAEMARHMM(BaseModel):
    def __init__(self,
			 	 data_loader: DataLoaderCollection,
				 config: GlobalConfig,
				 device: torch.device,
                 training_pipeline = 'cvae') -> None:
        super().__init__(data_loader, config, device)
        assert training_pipeline in ['end_to_end', 'marhmm', 'cvae'], "training must be 'end_to_end', 'marhmm' or 'cvae'"
        self.training_pipeline = training_pipeline
        self.data_loader = data_loader
        self.config = config
        self.device = device
        self.cvae = ConditionalVAE(
            data_loader=data_loader,
            config=config,
            device=device
        )
        self.marhmm = MARHMM(
            data_loader=data_loader,
            config=config,
            device=device,
            overwrite_obs_dim=self.cvae.latent_dim,
            get_latent_features=self.get_latent_representation
        )
        self.to(self.device)
    
    def forward(self, x: torch.Tensor, subject_ids: torch.Tensor, epoch: int) -> torch.Tensor:
        x_recon, mu, logvar, z = self.cvae.forward(x, subject_ids)
        cvae_loss, cvae_reg_loss = self.cvae.calculate_loss(x, x_recon, mu, logvar, z, epoch)
        if self.training_pipeline == 'cvae':
            return cvae_loss, cvae_reg_loss
        marhmm_loss = self.marhmm(mu)
        marhmm_reg_loss = self.marhmm.regularization_loss()
        if self.training_pipeline == 'marhmm':
            return marhmm_loss, marhmm_reg_loss
        if self.training_pipeline == 'end_to_end':
            return marhmm_loss + cvae_loss, marhmm_reg_loss + cvae_reg_loss
        raise ValueError(f"Unsupported training mode: {self.training_pipeline}")
    
    def prepare_for_training(self):
        self.train()
        for param in self.parameters():
            param.requires_grad = True
        if self.training_pipeline == 'cvae':
            for param in self.marhmm.parameters():
                param.requires_grad = False
        elif self.training_pipeline == 'marhmm':
            for param in self.cvae.parameters():
                param.requires_grad = False
    
    def prepare_for_inference(self):
        self.eval()
        for param in self.parameters():
            param.requires_grad = False
    
    def predict(self, x: torch.Tensor, subject_ids: torch.Tensor) -> torch.Tensor:
        mu = self.cvae.encode_to_latent(x, subject_ids)
        out = self.marhmm.predict(mu)
        return out
    
    def predict_gmm(self, x: torch.Tensor, subject_ids: torch.Tensor) -> torch.Tensor:
        y, likelihood, mu = self.cvae.predict_gmm_labels(x, subject_ids)
        return mu, y, likelihood
    
    def reinitialize_marhmm(self):
        self.marhmm = MARHMM(
            data_loader=self.data_loader,
            config=self.config,
            device=self.device,
            overwrite_obs_dim=self.cvae.latent_dim,
            get_latent_features=self.get_latent_representation
        )
        self.to(self.device)
        
    def reset(self):
        self.marhmm.reset()
        self.cvae.reset()
        self.prepare_for_training()
    
    def get_latent_representation(self, x: torch.Tensor, subject_ids: torch.Tensor) -> torch.Tensor:
        mu = self.cvae.encode_to_latent(x, subject_ids)
        return mu
    
    def regularization_loss(self, epoch: int) -> torch.Tensor:
        cvae_reg = self.cvae.regularization_loss(epoch)
        if self.training_pipeline == 'cvae':
            return cvae_reg
        marhmm_reg = self.marhmm.regularization_loss()
        if self.training_pipeline == 'marhmm':
            return marhmm_reg
        if self.training_pipeline == 'end_to_end':
            return cvae_reg + marhmm_reg
        raise ValueError(f"Unsupported training mode: {self.training_pipeline}")
    
    def __str__(self) -> str:
        return (
			f"CVAEMARHMM(states={self.num_states}, features={self.num_features}, "
            f"latent_dim={self.cvae.latent_dim}, hidden_dims={self.cvae.enc_hidden_dims}, "
            f"emb_dim={self.cvae.emb_dim})"
		)