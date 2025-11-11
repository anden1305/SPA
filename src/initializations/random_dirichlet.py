import torch
from src.models.base_model import BaseModel
from src.initializations.random_uniform import set_identity_covariance

def init_random_dirichlet(
    model: BaseModel,
    mean_std: float = 1.0,
    alpha: float = 1.0,
    self_transition_bias: float = 0.0,
) -> None:
    """Fully random means, unit covariance, Dirichlet-sampled π and A."""
    S, D = model.num_states, model.num_features
    device = next(model.parameters()).device

    # Random means (HMM) or AR coeffs (MAR-HMM)
    if hasattr(model, 'emission_mean'):
        model.emission_mean.copy_(mean_std * torch.randn(S, D, device=device))
    elif hasattr(model, 'coeffs'):
        L = len(model.lags)
        model.coeffs.copy_(mean_std * torch.randn(S, D, D * L, device=device))
        if hasattr(model, 'bias'):
            model.bias.copy_(0.1 * mean_std * torch.randn(S, D, device=device))

    set_identity_covariance(model)

    # Dirichlet-sampled transitions
    alpha_vec = torch.full((S,), alpha, device=device, dtype=model.initial_logits.dtype)
    pi = torch.distributions.Dirichlet(alpha_vec).sample()
    A = torch.stack([torch.distributions.Dirichlet(alpha_vec).sample() for _ in range(S)])

    model.initial_logits.copy_(pi.clamp_min(1e-12).log())
    model.transition_logits.copy_(A.clamp_min(1e-12).log())

    if self_transition_bias != 0.0:
        model.transition_logits.diagonal().add_(self_transition_bias)
