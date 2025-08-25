
from typing import Optional, Sequence
from torch import Tensor
from src.config.config import GlobalConfig
from src.data.data_loader import DataLoader
from src.models.base_model import MLModel
from src.training.trainer import Trainer
import matplotlib.pyplot as plt

class Visualizer:
    def __init__(self,
                 data_loader: DataLoader,
                 model: MLModel,
                 trainer: Trainer,
                 config: GlobalConfig):
        self.data_loader = data_loader
        self.model = model
        self.trainer = trainer
        self.global_config = config
        self.config = self.global_config.visualizer
    
    def visualize(self):
        if self.config.losses:
            self.plot_losses()

    def plot_pca_tripanel(
        self,
        x: Tensor,
        true_labels: Sequence[int],
        hmm_init_labels: Sequence[int],
        hmm_trained_labels: Sequence[int],
        *,
        subsample: int = 0,
        top_k: int = 4,
        out_dir: Optional[str] = None,
        filename_prefix: str = "hmm_tripanel",
        dpi: int = 160,
    ) -> list[str]:
        """Save tri-panel PCA plots: HMM-init vs HMM-trained vs True.

        Inputs
        - x: (T,D) or (B,T,D) tensor. If batched, it's flattened across batch.
        - true_labels, hmm_init_labels, hmm_trained_labels: length N label arrays.
        - subsample: if >0, uniformly subsample to at most this many points.
        - top_k: compute PCA up to this many components (max 4 by default).
        - out_dir: directory to save images. Defaults to results/run_name/plots.
        - filename_prefix: base prefix for filenames.
        Returns list of saved file paths.
        """
        try:
            import numpy as np
            import matplotlib.pyplot as _plt
        except Exception as e:
            raise RuntimeError("matplotlib and numpy required for plot_pca_tripanel") from e

        from itertools import combinations
        from pathlib import Path

        # Convert/flatten X to (N,D)
        if hasattr(x, "detach"):
            xt = x
            if xt.dim() == 3:
                B, T, D = xt.shape
                X_np = xt.reshape(B * T, D).detach().cpu().numpy()
            elif xt.dim() == 2:
                X_np = xt.detach().cpu().numpy()
            else:
                raise ValueError(f"x must be (T,D) or (B,T,D); got {tuple(xt.shape)}")
        else:
            X_np = np.asarray(x)
            if X_np.ndim != 2:
                raise ValueError(f"x numpy array must be 2D (N,D); got {X_np.shape}")

        true_arr = np.asarray(true_labels).ravel()
        init_arr = np.asarray(hmm_init_labels).ravel()
        trained_arr = np.asarray(hmm_trained_labels).ravel()

        N_total = X_np.shape[0]
        if not (len(true_arr) == len(init_arr) == len(trained_arr) == N_total):
            raise ValueError("Label lengths must match number of rows in x after flattening")

        # Subsample uniformly if requested
        if subsample and subsample > 0 and N_total > subsample:
            idx = np.linspace(0, N_total - 1, subsample).astype(int)
            X_plot = X_np[idx]
            true_plot = true_arr[idx]
            init_plot = init_arr[idx]
            trained_plot = trained_arr[idx]
        else:
            X_plot = X_np
            true_plot = true_arr
            init_plot = init_arr
            trained_plot = trained_arr

        # PCA via SVD (no sklearn dependency), up to top_k (<= D)
        K_req = int(max(2, min(top_k, X_plot.shape[1], 4)))
        Xc = X_plot - X_plot.mean(0, keepdims=True)
        U, Svals, _ = np.linalg.svd(Xc, full_matrices=False)
        # Components scores = U * S
        proj = (U[:, :K_req] * Svals[:K_req]) if Svals.size >= K_req else U[:, :Svals.size] * Svals[:Svals.size]

        # Color mapping helper
        palette = np.array(["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"])
        def _colors(arr):
            uniq = np.unique(arr)
            lut = {u: palette[i % len(palette)] for i, u in enumerate(uniq)}
            return np.array([lut[v] for v in arr])

        c_true = _colors(true_plot)
        c_init = _colors(init_plot)
        c_tr   = _colors(trained_plot)

        # Output directory
        if out_dir is None:
            base = Path(self.global_config.results_dir) / self.global_config.run_name / "plots"
        else:
            base = Path(out_dir)
        base.mkdir(parents=True, exist_ok=True)

        saved_paths: list[str] = []
        pairs = list(combinations(range(proj.shape[1]), 2))
        for a, b in pairs:
            fig, axes = _plt.subplots(1, 3, figsize=(11.4, 3.2), sharex=True, sharey=True)
            axes[0].scatter(proj[:, a], proj[:, b], c=c_init,  s=6, alpha=0.85, edgecolors='none')
            axes[1].scatter(proj[:, a], proj[:, b], c=c_tr,    s=6, alpha=0.85, edgecolors='none')
            axes[2].scatter(proj[:, a], proj[:, b], c=c_true,  s=6, alpha=0.85, edgecolors='none')
            axes[0].set_title(f'HMM init (PC{a+1} vs PC{b+1})')
            axes[1].set_title(f'HMM trained (PC{a+1} vs PC{b+1})')
            axes[2].set_title(f'True (PC{a+1} vs PC{b+1})')
            for ax in axes:
                ax.set_xlabel(f'PC{a+1}')
            axes[0].set_ylabel(f'PC{b+1}')
            _plt.tight_layout()
            out_path = base / f"{filename_prefix}_tripanel_pc{a+1}_pc{b+1}.png"
            fig.savefig(out_path.as_posix(), dpi=dpi)
            _plt.close(fig)
            saved_paths.append(out_path.as_posix())

        return saved_paths
    
    def plot_losses(self):
        losses = self.trainer.get_losses()
        plt.plot(losses)
        plt.title("Losses")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig(f"{self.global_config.results_dir}/{self.global_config.run_name}/losses.png")
