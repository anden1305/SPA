import torch

def calinski_harabasz_score(Z: torch.Tensor, assignments: torch.Tensor, centers: torch.Tensor) -> float:
    """Calinski-Harabasz clustering quality score."""
    K, N = centers.size(0), Z.size(0)
    if K <= 1 or N <= K:
        return float('-inf')
    
    overall_mean = Z.mean(dim=0, keepdim=True)
    within_scatter = (Z - centers[assignments]).pow(2).sum()
    cluster_counts = torch.bincount(assignments, minlength=K).float().unsqueeze(1)
    between_scatter = (cluster_counts * (centers - overall_mean).pow(2)).sum()
    
    return float((between_scatter / (K - 1)) / (within_scatter / (N - K) + 1e-12))