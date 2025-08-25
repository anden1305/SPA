
from typing import Optional, Sequence
from pathlib import Path
import numpy as np
from torch import Tensor
from src.config.config import GlobalConfig
from src.data.data_loader import DataLoader
from src.models.base_model import MLModel
from src.training.trainer import Trainer
from src.validation.validator import Validator
import matplotlib.pyplot as plt

class Visualizer:
    def __init__(self,
                 data_loader: DataLoader,
                 model: MLModel,
                 trainer: Trainer,
                 config: GlobalConfig,
                 validator: Validator):
        self.data_loader = data_loader
        self.model = model
        self.trainer = trainer
        self.global_config = config
        self.config = self.global_config.visualizer
        self.validator = validator
    
    def visualize(self):
        if self.config.losses:
            self.plot_losses()
        if self.config.pca_tripanel:
            self.plot_pca_tripanel()

    def plot_pca_tripanel(self):
        """Save tri-panel PCA plots comparing HMM-init, HMM-trained, and True labels.

        x may be (T,D) or (B,T,D). Labels must match flattened time length.
        """
        # Get data and predictions
        x, y_true = self.data_loader.get_all_data()
        preds = self.validator.get_predictions() if (self.validator is not None and hasattr(self.validator, "get_predictions")) else {}
        if not preds:
            return []
        epochs = sorted(preds.keys())
        init_seq = preds[epochs[0]]
        pred_seq = preds[epochs[-1]]

        # Flatten features to (N,D)
        xt = x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
        X = xt.reshape(-1, xt.shape[-1]) if xt.ndim >= 2 else None
        if X is None or X.ndim != 2:
            raise ValueError(f"x must be (T,D) or (B,T,D); got {xt.shape}")

        # Flatten labels to (N,)
        to_np = lambda a: (a.detach().cpu().numpy() if hasattr(a, "detach") else np.asarray(a)).reshape(-1)
        true_arr = to_np(y_true)
        init_arr = to_np(init_seq)
        trained_arr = to_np(pred_seq)
        if not (len(true_arr) == len(init_arr) == len(trained_arr) == X.shape[0]):
            raise ValueError("Label lengths must match number of rows in x after flattening")

        # PCA via SVD (up to 4 comps)
        K = min(4, max(2, X.shape[1]))
        U, S, _ = np.linalg.svd(X - X.mean(0, keepdims=True), full_matrices=False)
        proj = U[:, :K] * S[:K]

        # Colors and titles
        palette = np.array(["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"])
        def colors(a):
            u = np.unique(a)
            lut = {v: palette[i % len(palette)] for i, v in enumerate(u)}
            return np.array([lut[v] for v in a])
        cols = [colors(init_arr), colors(trained_arr), colors(true_arr)]
        titles = ["HMM init", "HMM trained", "True"]

        # Output dir
        base = Path(self.global_config.results_dir) / self.global_config.run_name / "plots"
        base.mkdir(parents=True, exist_ok=True)

        saved: list[str] = []
        from itertools import combinations
        for a, b in combinations(range(proj.shape[1]), 2):
            fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.2), sharex=True, sharey=True)
            for ax, c, t in zip(axes, cols, titles):
                ax.scatter(proj[:, a], proj[:, b], c=c, s=6, alpha=0.85, edgecolors="none")
                ax.set_title(f"{t} (PC{a+1} vs PC{b+1})")
                ax.set_xlabel(f"PC{a+1}")
            axes[0].set_ylabel(f"PC{b+1}")
            fig.tight_layout()
            out_path = base / f"hmm_tripanel_pc{a+1}_pc{b+1}.png"
            fig.savefig(out_path.as_posix(), dpi=160)
            plt.close(fig)
            saved.append(out_path.as_posix())

        return saved
    
    def plot_losses(self) -> None:
        losses = self.trainer.get_losses()
        plt.plot(losses)
        plt.title("Losses")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig(f"{self.global_config.results_dir}/{self.global_config.run_name}/losses.png")