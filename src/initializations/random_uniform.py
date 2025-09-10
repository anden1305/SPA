import torch
from src.models.base_model import BaseModel

def init_random_uniform(model: BaseModel, coeff_std: float = 1.0, jitter_std: float = 0.0, var_init: float = 1.0) -> None:
    """Fully random means or AR coeffs, unit covariance or scaled variance, uniform π and A."""
    S, D = model.num_states, model.num_features
    device = next(model.parameters()).device  # Use first parameter's device

    # Initialize means (for standard HMM) or AR coefficients (for MAR-HMM)
    if hasattr(model, 'emission_mean'):
        # Standard HMM: random means
        means = coeff_std * torch.randn(S, D, device=device)
        if jitter_std > 0:
            means += jitter_std * torch.randn_like(means)
        model.emission_mean.copy_(means)
    elif hasattr(model, 'coeffs'):
        # MAR-HMM: random AR coefficients
        L = len(model.lags)  # Assuming lags attribute exists
        coeffs = coeff_std * torch.randn(S, D, D * L, device=device)
        if jitter_std > 0:
            coeffs += jitter_std * torch.randn_like(coeffs)
        model.coeffs.copy_(coeffs)

    # Initialize covariance or variance
    if hasattr(model, 'emission_logvar') or hasattr(model, 'emission_cholesky_raw'):
        set_identity_covariance(model)  # Use existing function for standard HMM
    elif hasattr(model, 'log_var'):
        # MAR-HMM: initialize log_var to represent scaled variance
        target_var = max(var_init, 1e-6)  # Avoid numerical issues
        log_var = torch.full((S, D), torch.log(torch.tensor(target_var, device=device)), device=device)
        if jitter_std > 0:
            log_var += jitter_std * torch.randn_like(log_var)
        model.log_var.copy_(log_var)

    # Uniform initial and transition probabilities
    model.initial_logits.data.uniform_(-1.0, 1.0)
    model.transition_logits.data.uniform_(-1.0, 1.0)

def set_identity_covariance(model: BaseModel) -> None:
    """Set covariance to identity for any covariance type (standard HMM only)."""
    if model.covariance_type == "diag":
        model.emission_logvar.zero_() 
    elif model.covariance_type == "full":
        D = model.num_features
        device = next(model.parameters()).device
        raw = torch.zeros_like(model.emission_cholesky_raw)
        target = max(1.0 - float(model.jitter), 1e-6)
        diag_val = torch.log(torch.expm1(torch.tensor(target, device=device)))
        raw[:, torch.arange(D, device=device), torch.arange(D, device=device)] = diag_val
        model.emission_cholesky_raw.copy_(raw)