
from pyparsing import Any
import torch


def compute_summary_statistics(x: torch.Tensor, y: torch.Tensor) -> dict[str, Any]:
    
    # target statistics
    classes = [_y.item() for _y in y.unique()]
    class_counts = {cls: (y == cls).sum().item() for cls in classes}
    x_means = {cls: x[y == cls].mean().item() for cls in classes}
    x_stds = {cls: x[y == cls].std().item() for cls in classes}

    # input statistics
    x_shape = x.shape
    x_mean = x.mean().item()
    x_std = x.std().item()

    return {
        "input_shape": x_shape,
        "input_mean": x_mean,
        "input_std": x_std,
        "target_classes": classes,
        "target_counts": class_counts,
        "target_means": x_means,
        "target_stds": x_stds,
    }