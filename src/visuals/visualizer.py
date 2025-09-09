import json
from typing import Any, Optional, Sequence
from pathlib import Path
from torch import Tensor
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from src.config.config import GlobalConfig
from src.data.data_loader import DataLoader
from src.helpers.align_labels import align_labels_hungarian
from src.models.base_model import BaseModel
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
        x, y = self.data_loader.get_all_data()
        x = x.detach().cpu().numpy()
        y = y.detach().cpu().numpy()
        if self.config.losses:
            self.__plot_losses(train_details)
        if self.config.pca_tripanel:
            self.__plot_pca_tripanel(train_details, x=x, y=y)
        if self.config.confusion_matrix:
            self.__plot_confusion_matrix(train_details=train_details, y=y)
    
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
        state_names = self.data_loader.dataset.get_state_names()

        counts = np.array([float(target_counts.get(c, 0.0)) if target_counts is not None else 0.0 for c in (target_classes or [])])
        means = np.array([float(target_means.get(c, np.nan)) if target_means is not None else np.nan for c in (target_classes or [])])
        stds = np.array([float(target_stds.get(c, np.nan)) if target_stds is not None else np.nan for c in (target_classes or [])])

        # Prefer human-readable `target_labels` when available and matching length to `target_classes`.
        if state_names and target_classes and len(state_names) == len(target_classes):
            labels = list(state_names)
        else:
            labels = [str(c) for c in (target_classes or [])]

        # Plot 1: Global input mean/std vs per-class means (if available)
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        if input_mean is not None and input_std is not None:
            ax1.axhline(input_mean, color='black', linestyle='--', label=f'Global mean: {input_mean:.3f}')
            ax1.axhspan(input_mean - input_std, input_mean + input_std, color='black', alpha=0.12, label=f'Global ±1 std: {input_std:.3f}')

        if target_classes:
            x = np.arange(len(target_classes))
            ax1.errorbar(x, means, yerr=stds, fmt='o', color='tab:blue', ecolor='tab:gray', capsize=4, label='Per-class mean ± std')
            ax1.set_xticks(x)
            ax1.set_xticklabels(labels, rotation=45, ha='right')

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
            ax2.set_xticklabels(labels, rotation=45, ha='right')
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
            labels_for_plot = list(labels)
            for idx, (m, s) in enumerate(zip(means, stds)):
                # Use index-based deterministic seed so label type doesn't matter
                if np.isnan(m) or np.isnan(s) or s <= 0:
                    # fallback: show as single point
                    samp = np.array([m]) if not np.isnan(m) else np.array([0.0])
                else:
                    # sample up to 200 points but keep deterministic by using index as seed
                    rng = np.random.default_rng(idx + 1)
                    # choose sample size based on available counts (aligned by index)
                    count_est = int(counts[idx]) if counts.size > idx else 50
                    samp = rng.normal(loc=m, scale=s, size=min(200, max(20, count_est)))
                samples.append(samp)

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
        # human-readable state names (may be None)
        state_names = self.data_loader.dataset.get_state_names()

        ed_mat = np.asarray(dv.get("pairwise_energy", []), dtype=float)
        if ed_mat.size == 0:
            return
        states = list(dv.get("states", range(ed_mat.shape[0])))
        # If state_names provided and matches number of states, use them; otherwise fallback to integer labels
        if state_names and len(state_names) == len(states):
            state_labels = list(state_names)
        else:
            state_labels = [str(s) for s in states]
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
                # Use provided human-readable label when available
                label = state_labels[i]
                display_label = f"{label}\nmean ED={per_state_mean[i]:.2f}"
                ax.text(x[i]*1.15, y[i]*1.15, display_label,
                        ha='center', va='center', fontsize=11, fontweight='bold')

            # Draw lines between states, colored by ED
            norm = plt.Normalize(vmin=np.min(ed_mat[~np.eye(n_states, dtype=bool)]),
                               vmax=np.max(ed_mat[~np.eye(n_states, dtype=bool)]))

            for i in range(n_states):
                for j in range(i+1, n_states):
                    # Calculate midpoint with slight curve for visibility
                    mid_x = (x[i] + x[j]) / 2 * 0.8  # Pull toward center
                    mid_y = (y[i] + y[j]) / 2 * 0.8

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

            # Add a legend mapping colors to state labels
            legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.tab10(i % 10),
                                          label=state_labels[i], markersize=8)
                               for i in range(n_states)]
            ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=min(4, n_states), title='States')

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
    
    def __plot_confusion_matrix(self, train_details: TrainDetails, y: Tensor):
        
        init_arr = train_details.get_initial_predictions()
        trained_arr = train_details.get_trained_predictions()

        try:
            init_arr = align_labels_hungarian(y, init_arr)
            trained_arr = align_labels_hungarian(y, trained_arr)
        except Exception as e:
            print(f"Warning: Could not align labels for confusion matrix due to: {e}")
        init_cm = confusion_matrix(y, init_arr, normalize='true')
        pred_cm = confusion_matrix(y, trained_arr, normalize='true')

        # Determine human-readable labels if available
        state_names = self.data_loader.dataset.get_state_names()
        labels = None
        # Try to infer labels length from confusion matrix shape
        cm_size = init_cm.shape[0]
        if state_names and len(state_names) == cm_size:
            labels = list(state_names)
        else:
            labels = [str(i) for i in range(cm_size)]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        sns.heatmap(init_cm, ax=axes[0], annot=True, fmt=".2g", cmap="Blues", xticklabels=labels, yticklabels=labels)
        axes[0].set_title("Initial Confusion Matrix")
        axes[0].set_xlabel("Predicted")
        axes[0].set_ylabel("True")

        sns.heatmap(pred_cm, ax=axes[1], annot=True, fmt=".2g", cmap="Blues", xticklabels=labels, yticklabels=labels)
        axes[1].set_title("Trained Confusion Matrix")
        axes[1].set_xlabel("Predicted")
        axes[1].set_ylabel("True")

        plt.tight_layout()
        plt.savefig(train_details.get_path() / "plots" / "confusion_matrices.png")
        plt.close()

    def __plot_pca_tripanel(self, train_details: TrainDetails, x: Tensor, y: Tensor):
        """Save tri-panel PCA plots comparing HMM-init, HMM-trained, and True labels."""
        
        init_arr = train_details.get_initial_predictions()
        trained_arr = train_details.get_trained_predictions()

        X = x.reshape(-1, x.shape[-1]) if x.ndim >= 2 else None
        if X is None or X.ndim != 2:
            raise ValueError(f"x must be (T,D) or (B,T,D); got {x.shape}")
        
        try:
            init_arr = align_labels_hungarian(y, init_arr)
            trained_arr = align_labels_hungarian(y, trained_arr)
        except Exception as e:
            print(f"Warning: Could not align labels for PCA tripanel due to: {e}")
        
        if not (len(y) == len(init_arr) == len(trained_arr) == X.shape[0]):
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

        cols = [colors(init_arr), colors(trained_arr), colors(y)]
        titles = ["HMM init", "HMM trained", "True"]
        arrays = [init_arr, trained_arr, y]

        # human-readable state names (if available)
        state_names = self.data_loader.dataset.get_state_names()

        saved: list[str] = []
        for a, b in combinations(range(proj.shape[1]), 2):
            fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True, sharey=True)
            for ax, arr, c, t in zip(axes, arrays, cols, titles):
                # Create scatter plot
                scatter = ax.scatter(proj[:, a], proj[:, b], c=c, s=6, alpha=0.85, edgecolors="none")
                ax.set_title(f"{t} (PC{a+1} vs PC{b+1})")
                ax.set_xlabel(f"PC{a+1}")
                
                # Add legend with state/class names when possible
                unique_labels = np.unique(arr)
                # Build label text: prefer state_names if length matches, else fallback
                if state_names is not None and len(state_names) >= unique_labels.size:
                    label_texts = [state_names[int(label)] for label in unique_labels]
                else:
                    label_texts = [f"State {label}" for label in unique_labels]

                legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                  markerfacecolor=palette[i % len(palette)],
                                  label=label_texts[i], markersize=8)
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

    