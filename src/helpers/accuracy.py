

import torch
import numpy as np

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    assert y_true.shape == y_pred.shape, "Shape mismatch in accuracy calculation"
    correct = (y_true == y_pred).sum()
    total = y_true.size
    acc = correct / total if total > 0 else 0.0
    return round(acc, 4)