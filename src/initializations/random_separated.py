import torch
from src.models.base_model import BaseModel
from src.initializations.random_uniform import set_identity_covariance

def init_random_separated(model: BaseModel, spread: float = 2.0, jitter_std: float = 0.05) -> None:
    """Structured random initialization that separates state means."""
    S, D = model.num_states, model.num_features
    device = model.emission_mean.device
    
    # Create orthogonal directions
    if D >= S:
        dirs, _ = torch.linalg.qr(torch.randn(D, S, device=device), mode='reduced')
        dirs = dirs.T
    else:
        dirs = torch.randn(S, D, device=device)
        dirs = dirs / dirs.norm(dim=1, keepdim=True).clamp_min(1e-8)
    
    # Separated means (HMM) or AR coeffs (MAR-HMM)
    if hasattr(model, 'emission_mean'):
        means = spread * dirs + jitter_std * torch.randn(S, D, device=device)
        model.emission_mean.copy_(means)
    elif hasattr(model, 'coeffs'):
        coeffs = spread * dirs.view(S, D, 1).expand(-1, -1, D * len(model.lags)) + jitter_std * torch.randn(S, D, D*len(model.lags), device=device)
        model.coeffs.copy_(coeffs)
        if hasattr(model, 'bias'):
            model.bias.copy_(0.1 * spread * torch.randn(S, D, device=device))
    
    # Identity covariance and uniform transitions
    set_identity_covariance(model)
    # model.initial_logits.zero_() 
    # model.transition_logits.zero_()

    model.initial_logits.data.uniform_(-0.2, 0.2)
    model.transition_logits.data.uniform_(-0.2, 0.2)