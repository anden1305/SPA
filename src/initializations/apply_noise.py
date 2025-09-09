import torch
from src.models.base_model import BaseModel
@torch.no_grad()
def apply_noise_and_bias(
    model: BaseModel,
    mean_std: float,
    cov_noise_std: float,
    init_logits_std: float,
    self_transition_bias: float,
) -> None:
    """Apply small Gaussian noise to emissions and logits, and add optional self-transition bias."""
    if mean_std > 0:
        noise = torch.randn(model.emission_mean.shape, device=model.emission_mean.device)
        model.emission_mean.add_(mean_std * noise)
    if cov_noise_std > 0:
        if model.covariance_type == "diag":
            noise = torch.randn(model.emission_logvar.shape, device=model.emission_logvar.device)
            model.emission_logvar.add_(cov_noise_std * noise)
        elif model.covariance_type == "full":
            noise = cov_noise_std * torch.randn(model.emission_cholesky_raw.shape, device=model.emission_cholesky_raw.device)
            tril_mask = torch.tril(torch.ones_like(model.emission_cholesky_raw)).bool()
            model.emission_cholesky_raw[tril_mask] = model.emission_cholesky_raw[tril_mask] + noise[tril_mask]
    if init_logits_std > 0:
        model.initial_logits.add_(init_logits_std * torch.randn(model.initial_logits.shape, device=model.initial_logits.device))
        model.transition_logits.add_(init_logits_std * torch.randn(model.transition_logits.shape, device=model.transition_logits.device))
    if self_transition_bias != 0.0:
        model.transition_logits.diagonal().add_(float(self_transition_bias))