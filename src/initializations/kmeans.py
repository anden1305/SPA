import torch
from src.models.base_model import BaseModel


@torch.no_grad()
def init_kmeans(model: BaseModel, data: torch.Tensor, kmeans_iters: int, estimate_transitions: bool) -> None:
    """Initialize HMM emissions with k-means on provided data."""
    B, T, D = data.shape
    S = model.num_states
    if D != model.num_features:
        raise ValueError("obs_dim mismatch")
    
    flat = data.reshape(-1, D).to(model.device)
    centers, assign = run_kmeans(flat, S, int(kmeans_iters))
    
    # Set emission means directly from k-means centers
    model.emission_mean.copy_(centers)
    set_covariance_from_assignments(model, flat, assign)
    
    # Set transition parameters
    if estimate_transitions:
        d2_bt = (data.unsqueeze(2) - centers.view(1, 1, S, D)).pow(2).sum(-1)
        assign = d2_bt.argmin(-1).reshape(-1)
    
    set_transition_params(model, assign, B, T, S, estimate_transitions)

def run_kmeans(Z: torch.Tensor, K: int, iters: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Run k-means clustering."""
    centers = kmeans_plus_init(Z, K)
    
    if iters <= 0:
        return centers, torch.cdist(Z, centers).argmin(-1)

    for _ in range(iters):
        assign = torch.cdist(Z, centers).argmin(-1)
        counts = torch.bincount(assign, minlength=K)
        
        # Handle empty clusters
        empty = (counts == 0).nonzero(as_tuple=False).flatten()
        if empty.numel() > 0:
            farthest = torch.cdist(Z, centers).min(dim=1).values.topk(empty.numel()).indices
            centers[empty] = Z[farthest]
            assign = torch.cdist(Z, centers).argmin(-1)
            counts = torch.bincount(assign, minlength=K)

        # Update centers
        new_centers = torch.zeros_like(centers)
        new_centers.scatter_add_(0, assign.unsqueeze(1).expand(-1, Z.size(1)), Z)
        centers = new_centers / counts.clamp_min(1).unsqueeze(1)

    return centers, torch.cdist(Z, centers).argmin(-1)

def kmeans_plus_init(Z: torch.Tensor, K: int) -> torch.Tensor:
    """K-means++ initialization."""
    n, device = Z.size(0), Z.device
    centers = [Z[torch.randint(0, n, (1,), device=device)]]
    
    for _ in range(1, K):
        dists = torch.cdist(Z, torch.cat(centers)).min(dim=1).values.pow(2)
        probs = dists / dists.sum().clamp_min(1e-12)
        centers.append(Z[torch.multinomial(probs, 1)])
    
    return torch.cat(centers)

def set_covariance_from_assignments(model: BaseModel, X: torch.Tensor, assign: torch.Tensor) -> None:
    """Set covariance parameters from assignments."""
    if model.covariance_type == "diag":
        set_diag_covariance_from_assignments(model, X, assign)
    elif model.covariance_type == "full":
        set_full_covariance_from_assignments(model, X, assign)

def set_diag_covariance_from_assignments(model: BaseModel, X: torch.Tensor, assign: torch.Tensor) -> None:
    """Compute and set diagonal covariance from assignments (keeps current means)."""
    resid = X - model.emission_mean[assign]
    var = torch.zeros_like(model.emission_mean)
    var.scatter_add_(0, assign.view(-1, 1).expand_as(resid), resid.pow(2))
    counts = torch.bincount(assign, minlength=model.num_states).clamp_min(1)
    model.emission_logvar.copy_((var / counts.view(-1, 1)).clamp_min(1e-6).log())

def set_full_covariance_from_assignments(model: BaseModel, X: torch.Tensor, assign: torch.Tensor) -> None:
    """Set Cholesky-parameterized full covariances using softplus mapping.

    Writes lower-triangular raw parameters such that on reconstruction we use:
        L = tril(raw); diag(L) = softplus(diag(raw)) + jitter
    ensuring positive diagonals and positive-definite covariance Σ = L L^T.
    """
    S = model.num_states
    D = model.num_features
    raw = torch.zeros_like(model.emission_cholesky_raw)
    I = torch.eye(D, device=model.device, dtype=X.dtype)

    for k in range(S):
        Xk = X[assign == k]
        if Xk.size(0) <= 1:
            var_k = X.var(dim=0, unbiased=False).clamp_min(1e-6) if Xk.size(0) == 0 else Xk.var(dim=0, unbiased=False).clamp_min(1e-6)
            Ck = torch.diag(var_k) + model.jitter * I
        else:
            Xk_c = Xk - Xk.mean(0, keepdim=True)
            Ck = (Xk_c.T @ Xk_c) / max(Xk.size(0) - 1, 1) + model.jitter * I
        
        try:
            Lk = torch.linalg.cholesky(Ck)
        except RuntimeError:
            Lk = torch.linalg.cholesky(Ck + 1e-4 * I)
        
        raw[k] = torch.tril(Lk)
        # Inverse softplus
        target = (torch.diagonal(Lk) - model.jitter).clamp_min(1e-8)
        diag_raw = torch.log(torch.expm1(target))
        raw[k].diagonal().copy_(diag_raw)
    
    model.emission_cholesky_raw.copy_(raw)

def set_transition_params(model: BaseModel, assign: torch.Tensor, B: int, T: int, S: int, estimate_transitions: bool) -> None:
    """Set initial and transition probability parameters."""

    if estimate_transitions:
        z = assign.view(B, T)
        pi_counts = torch.bincount(z[:, 0], minlength=S).float() + 1e-3
        pi = pi_counts / pi_counts.sum()
        
        prev = z[:, :-1].reshape(-1)
        nxt = z[:, 1:].reshape(-1)
        joint_idx = prev * S + nxt
        trans_counts = torch.bincount(joint_idx, minlength=S * S).float().reshape(S, S) + 1e-3
        A = trans_counts / trans_counts.sum(-1, keepdim=True)
    else:
        pi = torch.full((S,), 1.0/S, device=model.device)
        A = torch.full((S, S), 1.0/S, device=model.device)
    
    model.initial_logits.copy_(pi.clamp_min(1e-12).log())
    model.transition_logits.copy_(A.clamp_min(1e-12).log())

