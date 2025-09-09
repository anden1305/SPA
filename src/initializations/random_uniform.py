import torch
from src.models.base_model import BaseModel

def init_random_uniform(model: BaseModel, mean_std: float = 1.0, jitter_std: float = 0.0) -> None:
    """Fully random means, unit covariance, uniform π and A."""
    S, D = model.num_states, model.num_features
    device = model.emission_mean.device
    
    # Random means
    means = mean_std * torch.randn(S, D, device=device)
    if jitter_std > 0:
        means += jitter_std * torch.randn_like(means)
    model.emission_mean.copy_(means)
    
    # Identity covariance and uniform transitions
    set_identity_covariance(model)
    model.initial_logits.zero_()
    model.transition_logits.zero_()

def set_identity_covariance(model: BaseModel) -> None:
    """Set covariance to identity for any covariance type."""
    if model.covariance_type == "diag":
        model.emission_logvar.zero_()
    elif model.covariance_type == "full":
        D = model.num_features
        device = model.emission_mean.device
        raw = torch.zeros_like(model.emission_cholesky_raw)
        target = max(1.0 - float(model.jitter), 1e-6)
        diag_val = torch.log(torch.expm1(torch.tensor(target, device=device)))
        raw[:, torch.arange(D, device=device), torch.arange(D, device=device)] = diag_val
        model.emission_cholesky_raw.copy_(raw)