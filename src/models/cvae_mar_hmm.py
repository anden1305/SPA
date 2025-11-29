
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
        self.cvae = ConditionalVAE(
            data_loader=data_loader,
            config=config,
            device=device
        )
        self.marhmm = MARHMM(
            data_loader=data_loader,
            config=config,
            device=device,
            overwrite_obs_dim=self.cvae.latent_dim
        )
        self.to(self.device)
    
    def forward(self, x: torch.Tensor, subject_ids: torch.Tensor) -> torch.Tensor:
        is_reshaped = False
        if len(x.shape) == 4:
            B, T, S, C = x.shape
            x = x.view(B * T, S, C).permute(0, 2, 1)
            subject_ids = subject_ids.view(B * T, -1)[:,0]
            is_reshaped = True
        x_recon, mu, logvar, _ = self.cvae.forward(x, subject_ids)
        cvae_loss = self.cvae.calculate_loss(x, x_recon, mu, logvar)
        if is_reshaped:
            mu = mu.view(B, T, -1)
        if self.training_pipeline == 'cvae':
            return cvae_loss
        marhmm_loss = self.marhmm(mu)
        if self.training_pipeline == 'marhmm':
            return marhmm_loss
        if self.training_pipeline == 'end_to_end':
            return marhmm_loss + cvae_loss
        raise ValueError(f"Unsupported training mode: {self.training_pipeline}")
    
    def predict(self, x: torch.Tensor, subject_ids: torch.Tensor) -> torch.Tensor:
        is_reshaped = False
        if len(x.shape) == 4:
            B, T, S, C = x.shape
            x = x.view(B * T, S, C).permute(0, 2, 1)
            subject_ids = subject_ids.view(B * T, -1)[:,0]
            is_reshaped = True
        mu = self.cvae.encode_to_latent(x, subject_ids)
        if is_reshaped:
            mu = mu.view(B, T, -1)
        out = self.marhmm.predict(mu)
        return out
    
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
    
    def reset(self):
        self.marhmm.reset()
        self.cvae.reset()
        self.prepare_for_training()
    
    def get_latent_representation(self, x: torch.Tensor, subject_ids: torch.Tensor) -> torch.Tensor:
        is_reshaped = False
        if len(x.shape) == 4:
            B, T, S, C = x.shape
            x = x.view(B * T, S, C).permute(0, 2, 1)
            subject_ids = subject_ids.view(B * T, -1)[:,0]
            is_reshaped = True
        mu = self.cvae.encode_to_latent(x, subject_ids)
        # if is_reshaped:
        #     mu = mu.view(B, T, -1)
        print("CVAEMARHMM LATENT output shape:", mu.shape)
        return mu
    
    def regularization_loss(self) -> torch.Tensor:
        cvae_reg = self.cvae.regularization_loss()
        marhmm_reg = self.marhmm.regularization_loss()
        if self.training_pipeline == 'cvae':
            return cvae_reg
        if self.training_pipeline == 'marhmm':
            return marhmm_reg
        if self.training_pipeline == 'end_to_end':
            return cvae_reg + marhmm_reg
        raise ValueError(f"Unsupported training mode: {self.training_pipeline}")
    
    def __str__(self) -> str:
        return (
			f"CVAEMARHMM(states={self.num_states}, features={self.num_features}, "
            f"latent_dim={self.cvae.latent_dim}, hidden_dims={self.cvae.hidden_dims}, "
            f"emb_dim={self.cvae.emb_dim})"
		)