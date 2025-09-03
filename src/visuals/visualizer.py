import json
from typing import Any, Optional, Sequence
from pathlib import Path
from torch import Tensor
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from src.config.config import GlobalConfig
from src.data.data_loader import DataLoader
from src.models.base_model import MLModel
from src.orchestrator import train_details
from src.orchestrator.train_details import TrainDetails
from src.training.trainer import Trainer
from src.validation.validator import Validator
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations


class Visualizer:
    def __init__(self,
                 data_loader: DataLoader,
                 config: GlobalConfig,
                 validator: Validator):
        self.data_loader = data_loader
        self.global_config = config
        self.config = self.global_config.visualizer
        self.validator = validator
        
    ####### GENERAL METHODS #######
    
    def visualize_data(self):
        path = Path(self.global_config.results_dir) / self.global_config.run_name / "plots"
        path.mkdir(parents=True, exist_ok=True)
        if self.global_config.visualizer.state_distinctness and self.global_config.validator.state_distinctness:
            self.__plot_state_distinctness(path=path)
        if self.global_config.visualizer.summary_statistics and self.global_config.validator.summary_statistics:
            self.__plot_summary_statistics(path=path)
    
    def visualize(self, train_details: TrainDetails):
        path = train_details.get_path() / "plots"
        path.mkdir(parents=True, exist_ok=True)
        if self.config.losses:
            self.__plot_losses(train_details)
        if self.config.pca_tripanel:
            self.__plot_pca_tripanel(train_details)
        if self.config.confusion_matrix:
            self.__plot_confusion_matrix(train_details=train_details)
    
    def visualize_runs(self, train_details: list[TrainDetails], validations: dict[str, Any]):
        path = Path(self.global_config.results_dir) / self.global_config.run_name / "plots"
        path.mkdir(parents=True, exist_ok=True)

        losses = [list(detail.losses.values()) for detail in train_details]
        final_losses = list(validations.get("loss", {}).values())
        nmis = list(validations.get("nmi", {}).values())
        cross_nmis = list(validations.get("cross_nmi", {}).values())

        self.__plot_run_losses(losses, path=path)
        
        if validations.get("nmi"):
            self.__plot_reliability(nmis, cross_nmis, final_losses, path=path)
        

    
    ####### HELPER METHODS #######
    
    def __plot_summary_statistics(self, path: Path):
        
        # load from validator
        dv = self.validator.get_data_validations()
        input_mean = dv.get("input_mean", None) # float
        input_std = dv.get("input_std", None) # float
        target_classes = dv.get("target_classes", None) # list[int]
        target_counts = dv.get("target_counts", None) # dict[int, float]
        target_means = dv.get("target_means", None) # dict[int, float]
        target_stds = dv.get("target_stds", None) # dict[int, float]

        counts = np.array([float(target_counts.get(c, 0.0)) if target_counts is not None else 0.0 for c in target_classes])
        means = np.array([float(target_means.get(c, np.nan)) if target_means is not None else np.nan for c in target_classes])
        stds = np.array([float(target_stds.get(c, np.nan)) if target_stds is not None else np.nan for c in target_classes])

        # Plot 1: Global input mean/std vs per-class means (if available)
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        if input_mean is not None and input_std is not None:
            ax1.axhline(input_mean, color='black', linestyle='--', label=f'Global mean: {input_mean:.3f}')
            ax1.axhspan(input_mean - input_std, input_mean + input_std, color='black', alpha=0.12, label=f'Global ±1 std: {input_std:.3f}')

        if target_classes:
            x = np.arange(len(target_classes))
            ax1.errorbar(x, means, yerr=stds, fmt='o', color='tab:blue', ecolor='tab:gray', capsize=4, label='Per-class mean ± std')
            ax1.set_xticks(x)
            ax1.set_xticklabels(target_classes, rotation=45, ha='right')

        ax1.set_title('Global input statistics vs Per-class means')
        ax1.set_ylabel('Feature value (mean)')
        ax1.legend(loc='best')
        fig1.tight_layout()
        fig1.savefig(path / 'summary_input_vs_class_means.png', dpi=200)
        plt.close(fig1)

        # Plot 2: Class counts with overlaid per-class mean (secondary axis)
        if target_classes:
            fig2, ax2 = plt.subplots(figsize=(10, 5))
            x = np.arange(len(target_classes))
            # bar plot of counts
            ax2.bar(x, counts, color='tab:orange', alpha=0.9)
            ax2.set_xticks(x)
            ax2.set_xticklabels(target_classes, rotation=45, ha='right')
            ax2.set_ylabel('Count')
            ax2.set_title('Per-class counts and means (with std shading)')

            # secondary axis for mean values
            ax2b = ax2.twinx()
            ax2b.plot(x, means, color='tab:blue', marker='o', linestyle='-', label='Per-class mean')
            ax2b.fill_between(x, means - stds, means + stds, color='tab:blue', alpha=0.15)
            ax2b.set_ylabel('Mean value')

            # annotate bars with proportions
            total = counts.sum() if counts.size > 0 else 1
            for xi, cnt in zip(x, counts):
                prop = cnt / total if total > 0 else 0.0
                ax2.text(xi, cnt + max(1.0, total * 0.01), f'{int(cnt)}\n({prop:.1%})', ha='center', va='bottom', fontsize=9)

            # Legends
            lines, labels = ax2b.get_legend_handles_labels()
            ax2.legend(lines, labels, loc='upper right')
            fig2.tight_layout()
            fig2.savefig(path / 'summary_counts_and_means.png', dpi=200)
            plt.close(fig2)

        # Plot 3: Distribution / variability summary — show means and stds as violin-like markers
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        if target_classes:
            # Create a synthetic 'per-class' distribution using mean ± std for visualization
            # Each class will be represented by samples drawn from N(mean, std) for plotting distributions
            samples = []
            labels_for_plot = []
            for c, m, s in zip(target_classes, means, stds):
                if np.isnan(m) or np.isnan(s) or s <= 0:
                    # fallback: show as single point
                    samp = np.array([m]) if not np.isnan(m) else np.array([0.0])
                else:
                    # sample up to 200 points but keep deterministic by using seed from hash
                    rng = np.random.default_rng(abs(hash(c)) % (2**32))
                    samp = rng.normal(loc=m, scale=s, size=min(200, max(20, int(counts[int(target_classes.index(c))]) if counts.size>0 else 50)))
                samples.append(samp)
                labels_for_plot.append(c)

            # Use violinplot for the synthetic distributions but fall back to boxplot markers when degenerate
            try:
                parts = ax3.violinplot(samples, showmeans=True, showextrema=False)
                for pc in parts['bodies']:
                    pc.set_alpha(0.6)
                ax3.set_xticks(np.arange(1, len(labels_for_plot)+1))
                ax3.set_xticklabels(labels_for_plot, rotation=45, ha='right')
                ax3.set_ylabel('Value')
                ax3.set_title('Per-class inferred distributions (from mean±std)')
            except Exception:
                # fallback to boxplot
                ax3.boxplot(samples)
                ax3.set_xticklabels(labels_for_plot, rotation=45, ha='right')
                ax3.set_title('Per-class boxplots (inferred)')

            # Annotate per-class mean/std as text above each violin
            for i, (m, s) in enumerate(zip(means, stds), start=1):
                ax3.text(i, np.nanmax(samples[i-1]) if samples[i-1].size>0 else (m if not np.isnan(m) else 0.0) , f'm={m:.2f}\nσ={s:.2f}', ha='center', va='bottom', fontsize=8)

        else:
            # No per-class info: show global mean/std if available
            if input_mean is not None and input_std is not None:
                ax3.bar([0], [1], color='tab:gray', alpha=0.0)
                ax3.text(0, 0.5, f'Global mean={input_mean:.3f}\nGlobal std={input_std:.3f}', ha='center', va='center')
                ax3.set_xticks([])
                ax3.set_title('Global input summary')

        fig3.tight_layout()
        fig3.savefig(path / 'summary_distributions.png', dpi=200)
        plt.close(fig3)


    def __plot_state_distinctness(self, path: Path):
        """Create a circular network visualization for pairwise Energy Distance between states.
        Annotates each state node with its mean pairwise ED to other states and displays Fisher trace.
        """

        # load from validator
        dv = self.validator.get_data_validations()

        ed_mat = np.asarray(dv.get("pairwise_energy", []), dtype=float)
        if ed_mat.size == 0:
            return
        states = list(dv.get("states", range(ed_mat.shape[0])))
        counts = dv.get("counts", {})
        # Per-state mean pairwise ED (exclude diagonal)
        n_states = len(states)
        if n_states < 2:
            return
        mask = ~np.eye(n_states, dtype=bool)
        per_state_mean = (ed_mat * mask).sum(axis=1) / mask.sum(axis=1)
        # Fisher trace, if available
        fisher_val = dv.get("fisher_trace", None)
        # Network in circular layout
        if n_states >= 3:  # Only create this plot if we have at least 3 states
            fig, ax = plt.subplots(figsize=(10, 9))
            
            # Create circular layout coordinates
            theta = np.linspace(0, 2*np.pi, n_states, endpoint=False)
            x = np.cos(theta)
            y = np.sin(theta)
            
            # Draw state nodes with mean ED annotation
            for i, state in enumerate(states):
                circle = plt.Circle((x[i], y[i]), 0.1, fill=True, 
                                   color=plt.cm.tab10(i % 10), 
                                   alpha=0.8)
                ax.add_artist(circle)
                ax.text(x[i]*1.15, y[i]*1.15, f"State {state}\nmean ED={per_state_mean[i]:.2f}", 
                      ha='center', va='center', fontsize=11, fontweight='bold')
            
            # Draw lines between states, colored by ED
            norm = plt.Normalize(vmin=np.min(ed_mat[~np.eye(n_states, dtype=bool)]), 
                               vmax=np.max(ed_mat[~np.eye(n_states, dtype=bool)]))
            
            for i in range(n_states):
                for j in range(i+1, n_states):
                    # Calculate midpoint with slight curve for visibility
                    mid_x = (x[i] + x[j])/2 * 0.8  # Pull toward center
                    mid_y = (y[i] + y[j])/2 * 0.8
                    
                    # Get points for curved line
                    t = np.linspace(0, 1, 50)
                    curve_x = (1-t)**2 * x[i] + 2*(1-t)*t * mid_x + t**2 * x[j]
                    curve_y = (1-t)**2 * y[i] + 2*(1-t)*t * mid_y + t**2 * y[j]
                    
                    # Draw the curve with color based on ED
                    line = ax.plot(curve_x, curve_y, '-', linewidth=2.5, 
                                 color=plt.cm.plasma(norm(ed_mat[i, j])),
                                 alpha=0.75)[0]
                    
                    # Add ED value text
                    text_x = mid_x * 1.2
                    text_y = mid_y * 1.2
                    ax.text(text_x, text_y, f"{ed_mat[i, j]:.2f}", 
                          ha='center', va='center', fontsize=9,
                          bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))
            
            ax.set_xlim(-1.3, 1.3)
            ax.set_ylim(-1.3, 1.3)
            ax.set_aspect('equal')
            ax.axis('off')
            
            # Add colorbar
            sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, shrink=0.75)
            cbar.set_label("Energy Distance (higher = better separation)")
            
            title = "Pairwise Energy Distance Network\n(Line color intensity shows separation strength)"
            if fisher_val is not None:
                title += f"\nFisher trace: {float(fisher_val):.3f}"
            plt.title(title)
            plt.tight_layout()
            plt.savefig(path / "pairwise_ed_network.png", dpi=200)
            plt.close(fig)
            

    def __remap_predictions_to_labels(self, y_true, y_pred):
        
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
    
    
    def __plot_confusion_matrix(self, train_details: TrainDetails):
        # plot confusion matrix between predictions and true labels
        _, labels = self.data_loader.get_all_data()
        init_arr = train_details.get_initial_predictions()
        trained_arr = train_details.get_trained_predictions()

        to_np = lambda a: (a.detach().cpu().numpy() if hasattr(a, "detach") else np.asarray(a)).reshape(-1)
        labels = to_np(labels)
        
        init_arr, _, _ = self.__remap_predictions_to_labels(labels, init_arr)
        trained_arr, _, _ = self.__remap_predictions_to_labels(labels, trained_arr)

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

    def __plot_pca_tripanel(self, train_details: TrainDetails):
        """Save tri-panel PCA plots comparing HMM-init, HMM-trained, and True labels."""
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
        
        init_arr, _, _ = self.__remap_predictions_to_labels(true_arr, init_arr)
        trained_arr, _, _ = self.__remap_predictions_to_labels(true_arr, trained_arr)

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
        arrays = [init_arr, trained_arr, true_arr]

        saved: list[str] = []
        for a, b in combinations(range(proj.shape[1]), 2):
            fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True, sharey=True)
            for ax, arr, c, t in zip(axes, arrays, cols, titles):
                # Create scatter plot
                scatter = ax.scatter(proj[:, a], proj[:, b], c=c, s=6, alpha=0.85, edgecolors="none")
                ax.set_title(f"{t} (PC{a+1} vs PC{b+1})")
                ax.set_xlabel(f"PC{a+1}")
                
                # Add legend with state/class numbers
                unique_labels = np.unique(arr)
                legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                  markerfacecolor=palette[i % len(palette)], 
                                  label=f'State {label}', markersize=8) 
                                  for i, label in enumerate(unique_labels)]
                ax.legend(handles=legend_elements, loc='best', title='States')
                
            axes[0].set_ylabel(f"PC{b+1}")
            fig.tight_layout()
            out_path = train_details.get_path() / "plots" / f"hmm_tripanel_pc{a+1}_pc{b+1}.png"
            fig.savefig(out_path.as_posix(), dpi=160)
            plt.close(fig)
            saved.append(out_path.as_posix())

        return saved

    def __plot_losses(self, train_details: TrainDetails) -> None:
        losses = list(train_details.losses.values())
        plt.plot(losses)
        plt.title("Losses")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig(train_details.get_path() / "plots" / "losses.png")
        plt.close()

    def __plot_run_losses(self, losses: list[list[float]], path: Path) -> None:
        for i, run_losses in enumerate(losses):
            plt.plot(run_losses, label=f"Run {i+1}")
        plt.title("Losses")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(path / "losses.png")
        plt.close()

    def __plot_reliability(self, nmis: list[float], cross_nmis: list[float], losses: list[float], path: Path):
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

    