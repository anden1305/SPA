"""Train HMM on precomputed synthetic data (unsupervised, single sequence).

Usage:
	python scripts/train_hmm.py --data-path data/synthetic_data/test --epochs 5 --num-states 4

Assumes folder contains:
	eeg.npy              (shape: C x T or T, or (T,C) after loading we arrange)
	labels.npy           (ignored for unsupervised training)
	config_copy.yml      (meta)

No validation / splitting (pure unsupervised). The entire sequence is one batch.
"""

from __future__ import annotations

import sys, pathlib, argparse, random
import numpy as np
import torch

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from scr.data.synthetic_loader import SyntheticLoader
from scr.data.synthetic_dataset import SyntheticDataset
from scr.preprocessing.collapse_dimensions import CollapseDimensions
from scr.preprocessing.fft import FFT
import numpy as np
import matplotlib.pyplot as plt    


from scr.data.synthetic_dataset import SyntheticDataset
from scr.models.hmm import HMM

def parse_args():
	P = argparse.ArgumentParser(description="Train HMM on synthetic EEG")
	P.add_argument("--data-path", type=pathlib.Path, required=True, default="data\\synthetic_data\\test", help="Synthetic dataset directory (contains eeg.npy, labels.npy, config_copy.yml)")
	P.add_argument("--epochs", type=int, default=150)
	P.add_argument("--num-states", type=int, default=4)
	P.add_argument("--lr", type=float, default=1e-2)
	P.add_argument("--seed", type=int, default=0)
	P.add_argument("--save", type=pathlib.Path, default=None, help="Output file (.pt)")
	P.add_argument("--normalize", action="store_true", help="Per-feature z-score over time")
	P.add_argument("--grad-clip", type=float, default=None, help="Gradient norm clip (optional)")
	P.add_argument("--save-pred", type=pathlib.Path, default=None, help="Optional .npy path to save decoded Viterbi state sequence")
	P.add_argument("--init", choices=["default","kmeans"], default="kmeans", help="Parameter init: default random or kmeans data-driven")
	P.add_argument("--kmeans-iters", type=int, default=1, help="K-means refinement iterations when --init kmeans")
	P.add_argument("--no-estimate-transitions", action="store_true", help="When using kmeans init, do not estimate pi/transition from clusters")
	P.add_argument("--scatter2d", action="store_true", default=True, help="Generate 2D PCA scatter (true vs predicted)")
	P.add_argument("--scatter-subsample", type=int, default=10000, help="Max number of time points to plot (uniform subsample)")
	P.add_argument("--plot-dir", type=pathlib.Path, default=pathlib.Path("results/training"), help="Directory to save plots")
	return P.parse_args()


def _normalized_mutual_info(preds: np.ndarray, labels: np.ndarray, eps: float = 1e-12) -> float:
	"""Compute NMI (arithmetic mean version like sklearn) between two integer labelings.

	NMI = 2 * I(U;V) / (H(U)+H(V)), where I is mutual information and H are entropies.
	"""
	preds = np.asarray(preds).ravel()
	labels = np.asarray(labels).ravel()
	if preds.shape[0] != labels.shape[0]:
		return float('nan')
	# Relabel to contiguous 0..K-1 for stability
	def _reindex(x):
		uniq, inv = np.unique(x, return_inverse=True)
		return inv, uniq.size
	p_int, k1 = _reindex(preds)
	l_int, k2 = _reindex(labels)
	# Contingency
	cont = np.zeros((k1, k2), dtype=np.float64)
	for i in range(p_int.size):
		cont[p_int[i], l_int[i]] += 1.0
	N = cont.sum()
	if N == 0:
		return float('nan')
	p_ij = cont / N
	p_i = p_ij.sum(axis=1, keepdims=True)
	p_j = p_ij.sum(axis=0, keepdims=True)
	# Mutual information
	with np.errstate(divide='ignore', invalid='ignore'):
		rat = p_ij / (p_i @ p_j)
		mask = p_ij > 0
		I = (p_ij[mask] * np.log(rat[mask] + eps)).sum()
	# Entropies
	H1 = - (p_i[p_i>0] * np.log(p_i[p_i>0])).sum()
	H2 = - (p_j[p_j>0] * np.log(p_j[p_j>0])).sum()
	den = H1 + H2
	if den <= eps:
		return 0.0
	return float(2 * I / den)


def seed_all(seed: int):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)


def main():
	args = parse_args()
	seed_all(args.seed)

	dataset = SyntheticDataset(str(args.data_path))  # Load synthetic dataset from specified path (pass as str to avoid Path + str issues)
	# Keep a handle to the transform to access window_size for plotting
	fft_transform = FFT({'window_size': 512})
	dimension_transform = CollapseDimensions({})

	dataloader = SyntheticLoader(
		dataset=dataset,
		batch_size=5120,
		shuffle=False,
		transforms=[
			fft_transform,
			dimension_transform
		]
	)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	# Grab (and cache) the single (full) batch produced by the synthetic loader + transforms
	# Loader yields a single tuple (X, Y) where X shape (T, D) after FFT+collapse.
	# batch_x_np, batch_y_np = next(iter(dataloader))  # (T,D), (T,) (labels unused for unsupervised)
	data = [(x,y) for x,y in dataloader]
	batch_x_np = np.concatenate([xx for xx, _ in data])
	batch_y_np = np.concatenate([yy for _, yy in data])
	
	# Determine final feature dimension after transforms (not the raw channel count)
	obs_dim = batch_x_np.shape[1] if batch_x_np.ndim == 2 else batch_x_np.shape[-1]
	model = HMM(num_states=dataset.n_stages, obs_dim=obs_dim, normalize_time=True, device=device)
	# Convert to torch tensor once; HMM forward can accept (T,D) and will add batch dim internally.
	batch_x = torch.from_numpy(batch_x_np).float()

	# Optional improved initialization
	if args.init == "kmeans":
		print(f"[Init] K-means (iters={args.kmeans_iters}, estimate_transitions={not args.no_estimate_transitions})")
		model.reset_parameters(batch_x, kmeans_iters=args.kmeans_iters, estimate_transitions=not args.no_estimate_transitions)
	else:
		model.reset_parameters_random(batch_x, mean_std=2, cov_noise_std=2, init_logits_std=2)

	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

	losses = []
	for epoch in range(1, args.epochs + 1):
		model.train()
		optimizer.zero_grad()
		logp = model(batch_x)
		loss = -logp.mean()
		loss.backward()
		if args.grad_clip is not None:
			torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
		optimizer.step()
		print(f"Epoch {epoch}/{args.epochs} | negNLL={loss.item():.4f}")
		losses.append(loss.item())

	# Plot loss curve
	plt.figure(figsize=(6,3))
	plt.plot(losses, linewidth=1.0)
	plt.xlabel('Epoch')
	plt.ylabel('negNLL')
	plt.title('Training Loss')
	plt.tight_layout()
	# i want the synthetic_data name in the save file
	plt.savefig(f"results/training/loss_curve_HMM_{args.data_path.name}_{args.epochs}.png")
	plt.show()

	# Decode and print only the predicted state sequence (nothing else)
	with torch.no_grad():
		preds_tensor = model.decode_viterbi(batch_x).squeeze(0).cpu()
	preds = preds_tensor.tolist()
	print(f"Predicted state sequence: {preds}")
	print(f"Batch Y (labels): {batch_y_np}")
	# NMI
	nmi = _normalized_mutual_info(preds_tensor.numpy(), batch_y_np)
	print(f"NMI: {nmi:.6f}")

	# ---------------- Simple 2D scatter (true vs predicted) ----------------
	if args.scatter2d:
		args.plot_dir.mkdir(parents=True, exist_ok=True)
		print("[Scatter2D] Building PCA projection and plotting true vs predicted labels...")
		from pathlib import Path as _P
		out_path = args.plot_dir / f"hmm_{args.data_path.name}_scatter2d.png"
		_ = model.plot_pca_scatter(
			batch_x,
			true_labels=batch_y_np,
			pred_labels=preds_tensor.numpy(),
			subsample=args.scatter_subsample,
			out_path=str(out_path),
		)
		print(f"[Scatter2D] Saved {out_path}")

	# Optional silent saves
	if args.save_pred:
		args.save_pred.parent.mkdir(parents=True, exist_ok=True)
		np.save(args.save_pred, np.array(preds, dtype=int))
	if args.save:
		args.save.parent.mkdir(parents=True, exist_ok=True)
		torch.save({
			"model_state": model.state_dict(),
			"args": vars(args),
			"obs_dim": dataset.n_features,
			"sequence_length": dataset.n_timesteps,
			"viterbi_path": preds,
		}, args.save)


if __name__ == "__main__":
	main()