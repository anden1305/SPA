
from typing import Any, Optional, Sequence
from pathlib import Path
import numpy as np
from torch import Tensor
from src.config.config import GlobalConfig
from src.data.data_loader import DataLoader
from src.models.base_model import MLModel
from src.orchestrator import train_details
from src.orchestrator.train_details import TrainDetails
from src.training.trainer import Trainer
from src.validation.validator import Validator
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
from scipy.optimize import linear_sum_assignment

class Visualizer:
    def __init__(self,
                 data_loader: DataLoader,
                 config: GlobalConfig):
        self.data_loader = data_loader
        self.global_config = config
        self.config = self.global_config.visualizer
    
    def visualize(self, train_details: TrainDetails):
        path = train_details.get_path() / "plots"
        path.mkdir(parents=True, exist_ok=True)
        if self.config.losses:
            self.plot_losses(train_details)
        if self.config.pca_tripanel:
            self.plot_pca_tripanel(train_details)
        if self.config.confusion_matrix:
            self.plot_confusion_matrix(train_details=train_details)
    
    def visualize_runs(self, train_details: list[TrainDetails], validations: dict[str, Any]):
        path = Path(self.global_config.results_dir) / self.global_config.run_name / "plots"
        path.mkdir(parents=True, exist_ok=True)

        losses = [list(detail.losses.values()) for detail in train_details]
        final_losses = list(validations.get("loss", {}).values())
        nmis = list(validations.get("nmi", {}).values())
        cross_nmis = list(validations.get("cross_nmi", {}).values())

        self.plot_run_losses(losses, path=path)

        # Visualize validations
        if validations.get("nmi"):
            self.plot_reliability(nmis, cross_nmis, final_losses, path=path)

    def remap_predictions_to_labels(self, y_true, y_pred):
        
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        # Encode labels to 0..C-1 for rows (true) and columns (pred)
        true_labels, y_true_enc = np.unique(y_true, return_inverse=True)
        pred_labels, y_pred_enc = np.unique(y_pred, return_inverse=True)
        n_true, n_pred = true_labels.size, pred_labels.size

        # Build contingency matrix M[i, j] = # of samples with true=i and pred=j
        M = np.zeros((n_true, n_pred), dtype=np.int64)
        np.add.at(M, (y_true_enc, y_pred_enc), 1)

        # Hungarian solves a *min*-cost problem; convert to cost to *maximize* matches
        # Pad to square to handle unequal numbers of classes robustly
        dim = max(n_true, n_pred)
        M_pad = np.zeros((dim, dim), dtype=np.int64)
        M_pad[:n_true, :n_pred] = M
        cost = M_pad.max() - M_pad

        row_ind, col_ind = linear_sum_assignment(cost)  # gives one-to-one assignment

        # Build mapping from predicted -> true classes, ignoring dummy rows/cols
        mapping = {}
        for r, c in zip(row_ind, col_ind):
            if r < n_true and c < n_pred:  # ignore padded dummies
                mapping[pred_labels[c]] = true_labels[r]

        # Apply mapping (unmapped predicted labels—if any—fall back to themselves)
        y_pred_remapped = np.array([mapping.get(p, p) for p in y_pred])

        acc = (y_pred_remapped == y_true).mean()
        return y_pred_remapped, mapping, acc
    
    def plot_confusion_matrix(self, train_details: TrainDetails):
        # plot confusion matrix between predictions and true labels
        _, labels = self.data_loader.get_all_data()
        init_arr = train_details.get_initial_predictions()
        trained_arr = train_details.get_trained_predictions()

        to_np = lambda a: (a.detach().cpu().numpy() if hasattr(a, "detach") else np.asarray(a)).reshape(-1)
        labels = to_np(labels)
        
        init_arr, _, _ = self.remap_predictions_to_labels(labels, init_arr)
        trained_arr, _, _ = self.remap_predictions_to_labels(labels, trained_arr)

        init_cm = confusion_matrix(labels, init_arr, normalize='true')
        pred_cm = confusion_matrix(labels, trained_arr, normalize='true')

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        sns.heatmap(init_cm, ax=axes[0], annot=True, fmt=".2g", cmap="Blues")
        axes[0].set_title("Initial Confusion Matrix")
        axes[0].set_xlabel("Predicted")
        axes[0].set_ylabel("True")

        sns.heatmap(pred_cm, ax=axes[1], annot=True, fmt=".2g", cmap="Blues")
        axes[1].set_title("Trained Confusion Matrix")
        axes[1].set_xlabel("Predicted")
        axes[1].set_ylabel("True")

        plt.tight_layout()
        plt.savefig(train_details.get_path() / "plots" / "confusion_matrices.png")
        plt.close()

    def plot_pca_tripanel(self, train_details: TrainDetails):
        """Save tri-panel PCA plots comparing HMM-init, HMM-trained, and True labels.

        x may be (T,D) or (B,T,D). Labels must match flattened time length.
        """
        # Get data and predictions
        x, y_true = self.data_loader.get_all_data()
        
        init_arr = train_details.get_initial_predictions()
        trained_arr = train_details.get_trained_predictions()

        # Flatten features to (N,D)
        xt = x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
        X = xt.reshape(-1, xt.shape[-1]) if xt.ndim >= 2 else None
        if X is None or X.ndim != 2:
            raise ValueError(f"x must be (T,D) or (B,T,D); got {xt.shape}")

        # Flatten labels to (N,)
        to_np = lambda a: (a.detach().cpu().numpy() if hasattr(a, "detach") else np.asarray(a)).reshape(-1)
        true_arr = to_np(y_true)
        
        init_arr, _, _ = self.remap_predictions_to_labels(true_arr, init_arr)
        trained_arr, _, _ = self.remap_predictions_to_labels(true_arr, trained_arr)

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
            out_path = train_details.get_path() / "plots" / f"hmm_tripanel_pc{a+1}_pc{b+1}.png"
            fig.savefig(out_path.as_posix(), dpi=160)
            plt.close(fig)
            saved.append(out_path.as_posix())

        return saved

    def plot_losses(self, train_details: TrainDetails) -> None:
        losses = list(train_details.losses.values())
        plt.plot(losses)
        plt.title("Losses")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig(train_details.get_path() / "plots" / "losses.png")
        plt.close()

    def plot_run_losses(self, losses: list[list[float]], path: Path) -> None:
        for i, run_losses in enumerate(losses):
            plt.plot(run_losses, label=f"Run {i+1}")
        plt.title("Losses")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(path / "losses.png")
        plt.close()

    def plot_reliability(self, nmis: list[float], cross_nmis: list[float], losses: list[float], path: Path):
        fig, ax1 = plt.subplots(figsize=(10, 5))

        x = np.arange(len(nmis))
        ax1.plot(x, nmis, label="NMI")
        ax1.plot(x, cross_nmis, label="Cross NMI")
        ax1.set_ylabel("NMI")
        ax1.legend(loc="upper left")

        ax2 = ax1.twinx()
        ax2.plot(x, losses, label="Final Loss", color="orange")
        ax2.set_ylabel("Loss")
        ax2.legend(loc="upper right")

        plt.title("Reliability over Runs")
        plt.xlabel("Run")
        plt.tight_layout()
        plt.savefig(path / "reliability_plot.png")
        plt.close()