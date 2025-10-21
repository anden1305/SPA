

from typing import Any
import torch
from src.data.data_loader import DataLoader
from src.data.data_loader_collection import DataLoaderCollection
from src.preprocessing.fft import FFT


def compute_frequency_statistics(x: torch.Tensor, y: torch.Tensor, data_loader: DataLoaderCollection) -> dict[str, Any]:
    """Compute average power per frequency for each state in `y`.

    Supports `x` shapes:
    - (B, F)
    - (B, T, F)

    Supports `y` shapes:
    - (B,)
    - (B, T)

    Returns a dict mapping state -> `torch.Tensor` of shape (F,) with the
    average (mean) across samples belonging to that state.
    If the dataset transforms do not include an `FFT` transform, returns an
    empty dict.
    """

    transforms = data_loader.get_transforms()
    has_fft = any(isinstance(t, FFT) for t in transforms)
    if not has_fft:
        # Perform an rFFT-based conversion from time-domain to frequency-domain.
        # Supported time-domain shapes:
        # - (B, T) -> rfft over last dim -> (B, Freq)
        # - (B, T, C) -> rfft over time dim -> (B, Freq, C) -> reshape to (B, C*Freq)
        if not isinstance(x, torch.Tensor):
            raise TypeError("x must be a torch.Tensor when performing FFT transform")
        if x.dim() == 2:
            # (B, T)
            Xc = torch.fft.rfft(x, dim=-1)
            power = (Xc.abs() ** 2).to(x.dtype)
            x = power
        elif x.dim() == 3:
            # (B, T, C) -> rfft over T -> (B, Freq, C) -> permute to (B, C, Freq) -> flatten
            Xc = torch.fft.rfft(x, dim=1)
            power = (Xc.abs() ** 2).to(x.dtype)
            power = power.permute(0, 2, 1)  # (B, C, Freq)
            x = power.reshape(power.shape[0], -1)
        else:
            raise ValueError("Unsupported x dimensions for FFT conversion. Expected 2 or 3 dims for time-domain input.")

    if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
        raise TypeError("x and y must be torch.Tensor instances")

    # Normalize inputs to sample-level pairs (N, F) and (N,)
    if x.dim() == 2:
        # (B, F)
        x_flat = x
        if y.dim() == 1:
            y_flat = y
        else:
            raise ValueError("y has time dimension but x does not")
    elif x.dim() == 3:
        # (B, T, F)
        B, T, F = x.shape
        if y.dim() == 1:
            # label per batch -> repeat for each time step
            if y.shape[0] != B:
                raise ValueError("Batch dimension mismatch between x and y")
            y_flat = y.unsqueeze(1).expand(-1, T).reshape(-1)
        elif y.dim() == 2:
            if y.shape != (B, T):
                raise ValueError("Shape mismatch: expected y shape (B,T) to match x")
            y_flat = y.reshape(-1)
        else:
            raise ValueError("Unsupported y dimensions for x with time dimension")
        x_flat = x.reshape(-1, F)
    else:
        raise ValueError("Unsupported x dimensions. Expected 2 or 3 dims")

    # Ensure y is 1-D and integer-like for grouping
    y_flat = y_flat.reshape(-1)

    # Convert y to a tensor we can compare (keep dtype)
    states = torch.unique(y_flat)

    stats: dict[str, Any] = {}
    for s in states:
        mask = y_flat == s
        key = str(int(s.item()))
        if mask.sum() == 0:
            stats[key] = [0.0] * x_flat.shape[1]
            continue
        selected = x_flat[mask]
        mean_vec = selected.mean(dim=0)
        try:
            stats[key] = mean_vec.detach().cpu().tolist()
        except Exception:
            stats[key] = mean_vec.tolist()
    out = {'frequency_statistics': stats}
    return out
