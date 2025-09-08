import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


def _as_numpy_1d_int(arr):
  """Return a 1D numpy int64 array from numpy, torch, or sequence inputs."""
  if isinstance(arr, np.ndarray):
    return arr.ravel().astype(np.int64)
  if isinstance(arr, torch.Tensor):
    return arr.flatten().cpu().numpy().astype(np.int64)
  # allow python lists/tuples
  a = np.asarray(arr)
  return a.ravel().astype(np.int64)


def align_labels_hungarian(y_true, y_pred):
  """
  Map predicted labels to true labels to maximize accuracy using the Hungarian algorithm.

  y_true, y_pred: 1D arrays/tensors of shape (N,), integer labels (not necessarily 0-based or contiguous).
  Returns:
    y_pred_aligned: predictions after optimal remapping (same type as input y_pred)
    mapping: dict {pred_label -> true_label}
    acc: float accuracy after remapping
  """
  # Convert to numpy 1D int arrays for SciPy/processing
  y_true_np = _as_numpy_1d_int(y_true)
  y_pred_np = _as_numpy_1d_int(y_pred)

  if y_true_np.size != y_pred_np.size:
    raise ValueError("y_true and y_pred must have same length")

  # Unique labels
  true_ids = np.unique(y_true_np)
  pred_ids = np.unique(y_pred_np)

  if true_ids.size != pred_ids.size:
    raise ValueError("Number of classes differ; 1–1 mapping requires same count.")

  K = true_ids.size

  # Map raw labels -> [0..K-1] via searchsorted
  t_idx = np.searchsorted(true_ids, y_true_np)
  p_idx = np.searchsorted(pred_ids, y_pred_np)

  # Build confusion matrix
  cm = np.bincount(t_idx * K + p_idx, minlength=K * K).reshape(K, K)

  # Hungarian solves a min-cost problem; to maximize matches, minimize (max - C)
  cost = cm.max() - cm
  row_ind, col_ind = linear_sum_assignment(cost)

  # mapping: predicted label value -> true label value
  mapping = {int(pred_ids[j]): int(true_ids[i]) for i, j in zip(row_ind, col_ind)}

  # Apply mapping
  mapped = np.vectorize(lambda v: mapping[int(v)])(y_pred_np)

  # Return same type as input y_pred
  if isinstance(y_pred, torch.Tensor):
    y_pred_aligned = torch.from_numpy(mapped).to(y_pred.dtype)
  else:
    y_pred_aligned = mapped

  return y_pred_aligned
