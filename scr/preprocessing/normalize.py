import numpy as np

from scr.preprocessing.base_transform import BaseTransform


class Normalize(BaseTransform):
    """Per-channel normalization for arrays shaped (C, T).

    This transform expects a tuple (x, y) where x has shape (C, T).
    Each channel c in x[c, :] is normalized independently over time T.
    Returns (x_norm, y) with the same x shape (C, T).

    Parameters
    ----------
    params : dict
        Optional keys:
        - 'center': bool (default True)  -> subtract per-channel mean
        - 'scale' : bool (default True)  -> divide by per-channel std
        - 'eps'   : float (default 1e-8) -> numerical stability for std
    """

    def __init__(self, params: dict | None = None):
        if params is None:
            params = {}
        super().__init__(params=params)
        self.center: bool = bool(self.params.get('center', True))
        self.scale: bool = bool(self.params.get('scale', True))
        self.eps: float = float(self.params.get('eps', 1e-8))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if not isinstance(x, np.ndarray):
            x = np.asarray(x)
        if x.ndim != 2:
            raise ValueError(f"Normalize expects x with shape (C, T); got {x.shape}")
        
        # Compute per-channel statistics along time axis (axis=1)
        if self.center or self.scale:
            mean = x.mean(axis=1, keepdims=True) if self.center else 0.0
            if self.scale:
                std = x.std(axis=1, keepdims=True)
                std = np.maximum(std, self.eps)
            else:
                std = 1.0

            x = (x - mean) / std

        return x.astype(np.float32, copy=False)


__all__ = ["Normalize"]
