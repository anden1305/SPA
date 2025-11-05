import torch
from src.models.base_model import BaseModel
from src.helpers.calinski_harabasz import calinski_harabasz_score
from src.initializations.kmeans import kmeans_plus_init, run_kmeans, set_covariance_from_assignments, set_transition_params

@torch.no_grad()
def init_kmeans_pca(model: BaseModel, data: torch.Tensor, kmeans_iters: int, estimate_transitions: bool) -> None:
    """Initialize emissions via k-means in the best PCA subspace.

    Searches 3D subspaces among top 4 PCs, runs k-means, picks best by Calinski-Harabasz score.
    Estimates transition parameters from temporal assignments if requested.
    """
    B, T, D = data.shape
    S = model.num_states
    X = data.reshape(-1, D).to(model.device)
    
    if X.size(0) < S:
        raise ValueError("Not enough frames for k-means init in PCA space.")
    
    # PCA and subspace selection
    X_centered = X - X.mean(0, keepdim=True)
    _, Svals, Vh = torch.linalg.svd(X_centered, full_matrices=False)
    V = Vh.T
    
    num_components = min(4, V.size(1))
    if num_components < 3:
        raise ValueError(f"Need at least 3 principal components; got {V.size(1)}.")
    
    candidates = [[i, j, k] for i in range(num_components) 
                  for j in range(i + 1, num_components) 
                  for k in range(j + 1, num_components)]

    # Find best subspace
    best_score, best_assign = float('-inf'), None
    
    for dims in candidates:
        Z = (X_centered @ V[:, dims]) / Svals[dims].clamp_min(1e-8)
        centers, _ = run_kmeans(Z, S, kmeans_iters)
        final_assigments = torch.cdist(Z, centers).argmin(-1)
        score = calinski_harabasz_score(Z, final_assigments, centers)
        
        if score > best_score:
            best_score, best_assign = score, final_assigments

    if best_assign is None:
        raise RuntimeError("PCA subspace selection failed.")

    # Set parameters
    model.emission_mean.copy_(torch.stack([X[best_assign == k].mean(0) for k in range(S)]))
    set_covariance_from_assignments(model, X, best_assign)
    set_transition_params(model, best_assign, B, T, S, estimate_transitions)




