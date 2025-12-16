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
    
    # Set emission means (HMM) or AR coeffs (MAR-HMM)
    if hasattr(model, 'emission_mean'):
        model.emission_mean.copy_(centers)
    elif hasattr(model, 'coeffs'):
        # For MAR-HMM, use k-means centers to initialize AR coefficients for each state
        # This is a heuristic: we can't directly map centers to AR coeffs.
        # A simple approach is to set the bias term to the centers and initialize coeffs to zero.
        if hasattr(model, 'bias'):
            model.bias.copy_(centers)
        model.coeffs.zero_()

    set_covariance_from_assignments(model, flat, assign)
    
    # Set transition parameters
    if estimate_transitions:
        d2_bt = (data.unsqueeze(2) - centers.view(1, 1, S, D)).pow(2).sum(-1)
        assign = d2_bt.argmin(-1).reshape(-1)
    
    set_transition_params(model, assign, B, T, S, estimate_transitions)

    # New: estimate state-wise AR coefficients by ridge least squares using assignments
    if hasattr(model, 'coeffs'):
        assign_bt = assign.view(B, T)
        estimate_statewise_ar_coeffs(model, data.to(model.device), assign_bt)

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
    if hasattr(model, 'emission_mean'):
        resid = X - model.emission_mean[assign]
    elif hasattr(model, 'bias'):
        resid = X - model.bias[assign]
    else:
        resid = X
    var = torch.zeros_like(model.emission_mean if hasattr(model, 'emission_mean') else model.bias)
    var.scatter_add_(0, assign.view(-1, 1).expand_as(resid), resid.pow(2))
    counts = torch.bincount(assign, minlength=model.num_states).clamp_min(1)
    
    if hasattr(model, 'emission_logvar'):
        model.emission_logvar.copy_((var / counts.view(-1, 1)).clamp_min(1e-6).log())
    elif hasattr(model, 'log_var'):
        model.log_var.copy_((var / counts.view(-1, 1)).clamp_min(1e-6).log())

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


@torch.no_grad()
def estimate_statewise_ar_coeffs(model: BaseModel, data: torch.Tensor, assign_bt: torch.Tensor) -> None:
    """Estimate MARHMM AR coefficients and bias per state via ridge least squares on assigned frames.

    - data: (B,T,D)
    - assign_bt: (B,T) state index per frame
    Updates:
      model.coeffs: (S,D,D*L)
      model.bias:   (S,D)
    Uses model.lags, model.max_lag, and model.ridge.
    Also recomputes emission covariance from AR residuals to match the fitted dynamics.
    """
    if not hasattr(model, 'coeffs'):
        return
    # Ensure computations happen on the same device as model parameters during init
    param_device = next(model.parameters()).device
    data = data.to(param_device)
    assign_bt = assign_bt.to(param_device)

    B, T, D = data.shape
    S = model.num_states
    Ls = list(getattr(model, 'lags', []))
    if not Ls:
        return
    max_lag = int(getattr(model, 'max_lag', max(Ls)))
    L = len(Ls)
    DL = D * L

    device = param_device
    dtype_z = data.dtype
    # Build lagged design once
    lag_stack = torch.zeros((B, T, DL), device=device, dtype=dtype_z)
    for j, lag in enumerate(Ls):
        if lag < T:
            lag_stack[:, lag:, j * D:(j + 1) * D] = data[:, :-lag, :]

    param_dtype = next(model.parameters()).dtype
    coeffs = torch.zeros((S, D, DL), device=device, dtype=param_dtype)
    bias = torch.zeros((S, D), device=device, dtype=param_dtype)
    ridge = float(getattr(model, 'ridge', 0.0))

    # Mask out frames without full lags
    valid_mask = torch.ones((B, T), device=device, dtype=torch.bool)
    if max_lag > 0:
        valid_mask[:, :max_lag] = False

    for s in range(S):
        mask_s = (assign_bt == s) & valid_mask
        Ns = int(mask_s.sum().item())
        if Ns <= 0:
            continue
        Zs = lag_stack[mask_s]  # (Ns, DL)
        Ys = data[mask_s]       # (Ns, D)
        if Zs.ndim != 2 or Zs.shape[0] < 2:
            continue
        # Augment with ones to fit bias
        ones = torch.ones((Zs.shape[0], 1), device=device, dtype=dtype_z)
        Zs_aug = torch.cat([Zs, ones], dim=1)  # (Ns, DL+1)
        XtX = Zs_aug.T @ Zs_aug  # (DL+1, DL+1)
        if ridge > 0.0:
            I = torch.eye(XtX.size(0), device=device, dtype=dtype_z)
            # Do not regularize the bias term (last diagonal)
            I[-1, -1] = 0.0
            XtX = XtX + ridge * I
        XtY = Zs_aug.T @ Ys  # (DL+1, D)
        try:
            W = torch.linalg.solve(XtX, XtY)  # (DL+1, D)
        except RuntimeError:
            W = torch.linalg.pinv(XtX) @ XtY
        A = W[:-1, :]  # (DL, D)
        b = W[-1, :]   # (D,)
        coeffs[s] = A.T.to(coeffs.dtype)  # (D, DL)
        bias[s] = b.to(bias.dtype)

    model.coeffs.copy_(coeffs.to(model.coeffs.dtype))
    if hasattr(model, 'bias'):
        model.bias.copy_(bias.to(model.bias.dtype))

    # Recompute emission covariance from AR residuals to align likelihood with fitted dynamics
    recompute_covariance_from_ar(model, data, assign_bt, lag_stack, valid_mask)


@torch.no_grad()
def recompute_covariance_from_ar(model: BaseModel, data: torch.Tensor, assign_bt: torch.Tensor,
                                 lag_stack: torch.Tensor | None = None, valid_mask: torch.Tensor | None = None) -> None:
    """Recompute emission covariance per state using residuals from current AR params.

    Accepts optionally precomputed lag_stack and valid_mask for efficiency.
    """
    if not hasattr(model, 'coeffs'):
        return
    # Align devices with model parameters during initialization
    param_device = next(model.parameters()).device
    data = data.to(param_device)
    assign_bt = assign_bt.to(param_device)
    if lag_stack is not None:
        lag_stack = lag_stack.to(param_device)
    if valid_mask is not None:
        valid_mask = valid_mask.to(param_device)

    B, T, D = data.shape
    S = model.num_states
    Ls = list(getattr(model, 'lags', []))
    L = len(Ls)
    DL = D * L
    device = param_device
    dtype_z = model.coeffs.dtype

    if lag_stack is None:
        lag_stack = torch.zeros((B, T, DL), device=device, dtype=dtype_z)
        for j, lag in enumerate(Ls):
            if lag < T:
                lag_stack[:, lag:, j * D:(j + 1) * D] = data[:, :-lag, :].to(dtype_z)
    else:
        lag_stack = lag_stack.to(device=device, dtype=dtype_z)

    if valid_mask is None:
        valid_mask = torch.ones((B, T), device=device, dtype=torch.bool)
        max_lag = int(getattr(model, 'max_lag', max(Ls)))
        if max_lag > 0:
            valid_mask[:, :max_lag] = False

    # Compute residuals by state
    for s in range(S):
        mask_s = (assign_bt == s) & valid_mask
        Ns = int(mask_s.sum().item())
        if Ns <= 0:
            continue
        Zs = lag_stack[mask_s]  # (Ns, DL)
        Ys = data[mask_s]       # (Ns, D)
        pred = Zs @ model.coeffs[s].T  # (Ns, D)
        if hasattr(model, 'bias'):
            pred = pred + model.bias[s].view(1, D)
        resid = (Ys - pred)

        if model.covariance_type == 'diag':
            var = resid.var(dim=0, unbiased=False).clamp_min(1e-6)  # (D,)
            if hasattr(model, 'log_var'):
                model.log_var[s].copy_(var.log().to(model.log_var.dtype))
            elif hasattr(model, 'emission_logvar'):
                model.emission_logvar[s].copy_(var.log().to(model.emission_logvar.dtype))
        elif model.covariance_type == 'full':
            resid_c = resid - resid.mean(0, keepdim=True)
            Ck = (resid_c.T @ resid_c) / max(Ns - 1, 1)
            I = torch.eye(D, device=device, dtype=dtype_z)
            Ck = Ck + model.jitter * I
            try:
                Lk = torch.linalg.cholesky(Ck)
            except RuntimeError:
                Lk = torch.linalg.cholesky(Ck + 1e-4 * I)
            raw = model.emission_cholesky_raw
            raw_k = torch.tril(Lk)
            target = (torch.diagonal(Lk) - model.jitter).clamp_min(1e-8)
            diag_raw = torch.log(torch.expm1(target))
            raw_k.diagonal().copy_(diag_raw)
            raw[s].copy_(raw_k.to(raw.dtype))

