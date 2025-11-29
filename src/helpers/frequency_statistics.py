

from typing import Any
import torch
from src.data.data_loader_collection import DataLoaderCollection


def compute_feature_statistics(x: torch.Tensor, y: torch.Tensor, data_loader: DataLoaderCollection, has_features: bool) -> dict[str, Any]:
    if not has_features:
        print('Skipping feature statistics calculation due to high dimensionality.')
        return {}
    if len(y.shape) == 3:
        y = y[:, :, 0]
    x_flat = x.reshape(-1, x.shape[-1])
    y_flat = y.reshape(-1)
    states = torch.unique(y_flat)

    means: dict[str, list[float]] = {}
    stds: dict[str, list[float]] = {}
    variances: dict[str, list[float]] = {}

    for s in states:
        mask = y_flat == s
        key = str(int(s.item()))
        if mask.sum() == 0:
            empty = [float('nan')] * x_flat.shape[1]
            means[key] = empty
            stds[key] = empty
            variances[key] = empty
            continue

        selected = x_flat[mask]
        mean_vec = selected.mean(dim=0)
        var_vec = selected.var(dim=0, unbiased=False)
        std_vec = torch.sqrt(var_vec)

        try:
            means[key] = mean_vec.detach().cpu().tolist()
            variances[key] = var_vec.detach().cpu().tolist()
            stds[key] = std_vec.detach().cpu().tolist()
        except Exception:
            means[key] = mean_vec.tolist()
            variances[key] = var_vec.tolist()
            stds[key] = std_vec.tolist()

    out = {
        'feature_statistics': {
            'amplitude': means,
            'amplitude_std': stds,
            'variance': variances,
        }
    }
    return out
