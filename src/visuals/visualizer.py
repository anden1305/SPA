from typing import Any, Optional, Sequence
from pathlib import Path
from torch import Tensor
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import plotly.graph_objects as go
from src.config.config import GlobalConfig
from src.data.data_loader import DataLoader
from src.helpers.align_labels import align_labels_hungarian
from src.orchestrator.train_details import TrainDetails
from src.validation.validator import Validator
from sklearn.metrics import confusion_matrix
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
        y = y.detach().cpu().numpy().flatten()
        if self.config.losses:
            self.__plot_losses(train_details)
        if self.config.pca_tripanel:
            self.__plot_pca_tripanel(train_details, x=x, y=y)
        if self.config.confusion_matrix:
            self.__plot_confusion_matrix(train_details=train_details, y=y)
            self.__plot_confusion_matrix_dynamic(train_details=train_details, y=y)
            self.__plot_metrics_over_epochs(train_details=train_details)
        if self.config.historic_values:
            self.__plot_historic_values(train_details)
    
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

    def __plot_confusion_matrix_dynamic(self, train_details: TrainDetails, y: np.ndarray, fps: float = 2.0):
        """Create interactive confusion matrix plot with slider and play controls showing evolution during training.
        
        Args:
            train_details: Training details containing predictions
            y: True labels
            fps: Frames per second for animation (default: 2.0)
        """
        if not train_details.predictions:
            return  # No predictions available
        
        y_true = y.flatten() if y.ndim > 1 else y
        labels = self.data_loader.dataset.get_state_names()
        if labels is None or len(labels) != len(np.unique(y_true)):
            labels = [str(i) for i in range(len(np.unique(y_true)))]
        
        out_dir = train_details.get_path() / "plots" 
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Get epochs that have predictions (sorted)
        epochs = sorted(train_details.predictions.keys())
        
        if len(epochs) == 0:
            return
        
        # Prepare confusion matrices for all epochs
        confusion_matrices = []
        for epoch in epochs:
            predictions = np.array(train_details.predictions[epoch])
            
            try:
                # Align predictions with true labels
                aligned_predictions = align_labels_hungarian(y_true, predictions)
            except Exception:
                aligned_predictions = predictions
            
            # Create confusion matrix (normalized)
            cm = confusion_matrix(y_true, aligned_predictions, normalize='true')
            confusion_matrices.append(cm)
        
        # Create interactive plotly figure with slider
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        # Initialize figure
        fig = go.Figure()
        
        # Add initial heatmap (first epoch)
        initial_cm = confusion_matrices[0]
        
        heatmap = go.Heatmap(
            z=initial_cm,
            x=labels,
            y=labels,
            colorscale='Blues',
            zmin=0,
            zmax=1,
            showscale=True,
            text=[[f'{val:.3f}' for val in row] for row in initial_cm],
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverongaps=False
        )
        
        fig.add_trace(heatmap)
        
        # Calculate frame duration from FPS
        frame_duration = int(1000 / fps)  # Convert FPS to milliseconds
        transition_duration = min(300, frame_duration // 2)  # Smooth transition
        
        # Create slider steps and frames for animation
        steps = []
        frames = []
        
        for i, epoch in enumerate(epochs):
            cm = confusion_matrices[i]
            
            # Create slider step
            step = dict(
                method="animate",
                args=[[f"frame{i}"], {
                    "frame": {"duration": frame_duration, "redraw": True},
                    "mode": "immediate",
                    "transition": {"duration": transition_duration}
                }],
                label=f"Epoch {epoch + 1}"
            )
            steps.append(step)
            
            # Get metrics for this epoch
            nmi_val = train_details.validations.get(epoch, {}).get('nmi', 'N/A')
            acc_val = train_details.validations.get(epoch, {}).get('accuracy', 'N/A')
            
            # Format metrics for display
            nmi_str = f"{nmi_val:.3f}" if isinstance(nmi_val, (int, float)) else str(nmi_val)
            acc_str = f"{acc_val:.3f}" if isinstance(acc_val, (int, float)) else str(acc_val)
            
            # Create animation frame
            frame = go.Frame(
                data=[go.Heatmap(
                    z=cm,
                    x=labels,
                    y=labels,
                    colorscale='Blues',
                    zmin=0,
                    zmax=1,
                    showscale=True,
                    text=[[f'{val:.3f}' for val in row] for row in cm],
                    texttemplate='%{text}',
                    textfont={"size": 10},
                    hoverongaps=False
                )],
                name=f"frame{i}",
                layout=go.Layout(
                    title_text=f"Confusion Matrix - Epoch {epoch + 1}<br>Run {train_details.run_number}<br>NMI: {nmi_str} | Accuracy: {acc_str}"
                )
            )
            frames.append(frame)
        
        # Add frames to figure
        fig.frames = frames
        
        # Add play/pause buttons with speed controls
        fig.update_layout(
            updatemenus=[
                {
                    # Main play/pause controls
                    "buttons": [
                        {
                            "args": [None, {
                                "frame": {"duration": frame_duration, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": transition_duration, "easing": "quadratic-in-out"}
                            }],
                            "label": "▶ Play",
                            "method": "animate"
                        },
                        {
                            "args": [[None], {
                                "frame": {"duration": 0, "redraw": True},
                                "mode": "immediate",
                                "transition": {"duration": 0}
                            }],
                            "label": "⏸ Pause",
                            "method": "animate"
                        }
                    ],
                    "direction": "left",
                    "pad": {"r": 10, "t": 87},
                    "showactive": False,
                    "type": "buttons",
                    "x": 0.1,
                    "xanchor": "right",
                    "y": 0.02,
                    "yanchor": "top"
                },
                {
                    # Speed controls
                    "buttons": [
                        {
                            "args": [None, {
                                "frame": {"duration": 2000, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 300}
                            }],
                            "label": "0.5x",
                            "method": "animate"
                        },
                        {
                            "args": [None, {
                                "frame": {"duration": 1000, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 200}
                            }],
                            "label": "1x",
                            "method": "animate"
                        },
                        {
                            "args": [None, {
                                "frame": {"duration": 500, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 100}
                            }],
                            "label": "2x",
                            "method": "animate"
                        },
                        {
                            "args": [None, {
                                "frame": {"duration": 250, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 50}
                            }],
                            "label": "4x",
                            "method": "animate"
                        }
                    ],
                    "direction": "left",
                    "pad": {"r": 10, "t": 50},
                    "showactive": False,
                    "type": "buttons",
                    "x": 0.4,
                    "xanchor": "right",
                    "y": 0.02,
                    "yanchor": "top"
                }
            ],
            sliders=[{
                "active": 0,
                "yanchor": "top",
                "xanchor": "left",
                "currentvalue": {
                    "font": {"size": 20},
                    "prefix": "Epoch: ",
                    "visible": True,
                    "xanchor": "right"
                },
                "transition": {"duration": 300, "easing": "cubic-in-out"},
                "pad": {"b": 10, "t": 50},
                "len": 0.9,
                "x": 0.1,
                "y": 0,
                "steps": steps
            }]
        )
        
        # Get initial metrics for the title
        initial_epoch = epochs[0]
        initial_nmi = train_details.validations.get(initial_epoch, {}).get('nmi', 'N/A')
        initial_acc = train_details.validations.get(initial_epoch, {}).get('accuracy', 'N/A')
        initial_nmi_str = f"{initial_nmi:.3f}" if isinstance(initial_nmi, (int, float)) else str(initial_nmi)
        initial_acc_str = f"{initial_acc:.3f}" if isinstance(initial_acc, (int, float)) else str(initial_acc)
        
        # Update main layout
        fig.update_layout(
            title=f"Confusion Matrix - Epoch {initial_epoch + 1}<br>Run {train_details.run_number}<br>NMI: {initial_nmi_str} | Accuracy: {initial_acc_str}",
            xaxis_title="Predicted State",
            yaxis_title="True State",
            width=650,
            height=650,
            font=dict(size=12)
        )
        
        # Reverse y-axis to match typical confusion matrix layout
        fig.update_yaxes(autorange="reversed")
        
        # Save as HTML
        html_path = out_dir / 'confusion_matrix_dynamic.html'
        fig.write_html(str(html_path))
        
        if self.global_config.verbose:
            print(f"📊 Interactive confusion matrix saved: {html_path}")
            print(f"   📽️  Use Play/Pause controls and speed buttons (0.5x to 4x)")
            print(f"   🎚️  Drag slider to navigate epochs manually")
            print(f"   ⚡ Animation speed: {fps:.1f} FPS ({frame_duration}ms per frame)")
            print(f"   📈 Evolution across {len(epochs)} epochs of training")

    def __plot_metrics_over_epochs(self, train_details: TrainDetails):
        """Create a line plot showing NMI and Accuracy over training epochs."""
        if not train_details.validations:
            return  # No validation data available
        
        out_dir = train_details.get_path() / "plots"
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract epochs and metrics
        epochs = sorted(train_details.validations.keys())
        nmi_values = []
        acc_values = []
        
        for epoch in epochs:
            validation_data = train_details.validations[epoch]
            nmi_val = validation_data.get('nmi', None)
            acc_val = validation_data.get('accuracy', None)
            
            nmi_values.append(nmi_val if isinstance(nmi_val, (int, float)) else None)
            acc_values.append(acc_val if isinstance(acc_val, (int, float)) else None)
        
        # Create matplotlib figure
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # Convert to 1-based epochs for display
        epoch_display = [e + 1 for e in epochs]
        
        # Plot NMI
        if any(v is not None for v in nmi_values):
            # Filter out None values for plotting
            valid_nmi = [(x, y) for x, y in zip(epoch_display, nmi_values) if y is not None]
            if valid_nmi:
                x_nmi, y_nmi = zip(*valid_nmi)
                ax.plot(x_nmi, y_nmi, 'o-', color='blue', linewidth=2, markersize=6, label='NMI')
        
        # Plot Accuracy
        if any(v is not None for v in acc_values):
            # Filter out None values for plotting
            valid_acc = [(x, y) for x, y in zip(epoch_display, acc_values) if y is not None]
            if valid_acc:
                x_acc, y_acc = zip(*valid_acc)
                ax.plot(x_acc, y_acc, 's-', color='red', linewidth=2, markersize=6, label='Accuracy')
        
        # Customize plot
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Metric Value', fontsize=12)
        ax.set_title(f'Training Metrics Over Epochs - Run {train_details.run_number}', fontsize=14)
        ax.set_ylim(0, 1)  # Metrics are typically 0-1
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)
        
        # Save as PNG
        png_path = out_dir / 'metrics_over_epochs.png'
        fig.tight_layout()
        fig.savefig(png_path, dpi=170, bbox_inches='tight')
        plt.close(fig)
        
        if self.global_config.verbose:
            print(f"📈 Metrics plot saved: {png_path}")
            print(f"   📊 Shows NMI and Accuracy evolution over {len(epochs)} epochs")

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
            
    def __plot_historic_values(self, train_details: TrainDetails):
        """Enhanced parameter-history visualization with specialized diagnostics.

        Saves static PNGs and (for <=2D params) optional HTML sliders.
        """
        hist = getattr(train_details, 'historic_values', {}) or {}
        if not hist:
            return
        base = train_details.get_path() / "plots" / "params"
        base.mkdir(parents=True, exist_ok=True)

        def _to_np(t):
            if t is None:
                return None
            if isinstance(t, torch.Tensor):
                return t.detach().cpu().numpy()
            return np.asarray(t)

        def _save(fig, name):
            fig.tight_layout()
            fig.savefig(base / name, dpi=170)
            plt.close(fig)

        def _quantile_ribbon(ax, data_2d, color='tab:blue', label=None):
            if data_2d.ndim != 2 or data_2d.shape[1] == 0:
                return
            q10 = np.nanquantile(data_2d, 0.10, axis=1)
            q50 = np.nanquantile(data_2d, 0.50, axis=1)
            q90 = np.nanquantile(data_2d, 0.90, axis=1)
            ax.fill_between(range(len(q10)), q10, q90, color=color, alpha=0.25)
            ax.plot(q50, color=color, lw=1.4, label=label or 'median (10–90%)')

        def _first_mid_last(E):
            if E <= 3:
                return list(range(E))
            return [0, E//2, E-1]

        model_type = self.global_config.model.type.lower()

        # Pass 1: generic plots
        for name, series in hist.items():
            if not series:
                continue
            sample = _to_np(series[0])
            if sample is None:
                continue
            E = len(series)

            # Scalars
            if sample.ndim == 0:
                vals = np.array([_to_np(s).item() for s in series])
                fig, ax = plt.subplots(figsize=(5,3))
                ax.plot(vals, lw=1.5)
                ax.set_title(name)
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Value')
                _save(fig, f"{name}.png")
                continue

            # Vectors
            if sample.ndim == 1:
                arr = np.stack([_to_np(s) for s in series], axis=0)
                N = arr.shape[1]
                fig, ax = plt.subplots(figsize=(7,4))
                if N <= 32:
                    ax.plot(arr)
                    ax.legend([f"i={i}" for i in range(N)], fontsize=6, ncol=4, frameon=False)
                else:
                    _quantile_ribbon(ax, arr, label='distribution')
                    var_idx = np.argsort(arr.var(axis=0))[-5:]
                    ax.plot(arr[:, var_idx], lw=1, alpha=0.85)
                    ax.legend(frameon=False)
                ax.set_title(name)
                ax.set_xlabel('Epoch'); ax.set_ylabel('Value')
                _save(fig, f"{name}.png")
                continue

            # Matrices
            if sample.ndim == 2:
                mats = [ _to_np(s) for s in series ]
                means = np.array([m.mean() for m in mats])
                norms = np.array([np.linalg.norm(m) for m in mats])
                fig, ax = plt.subplots(1,2, figsize=(9,3))
                ax[0].plot(means); ax[0].set_title(f"{name} mean")
                ax[1].plot(norms); ax[1].set_title(f"{name} Fro norm")
                for a in ax: a.set_xlabel('Epoch')
                _save(fig, f"{name}_stats.png")
                for idx in _first_mid_last(E):
                    fig, hx = plt.subplots(figsize=(4,4))
                    sns.heatmap(mats[idx], ax=hx, cmap='coolwarm', center=0 if np.abs(mats[idx]).max() < 5 else None)
                    hx.set_title(f"{name} e{idx}")
                    _save(fig, f"{name}_e{idx}.png")
                continue

            # Higher-order tensors
            flat = np.array([_to_np(s).ravel() for s in series])
            fig, ax = plt.subplots(figsize=(6,3.3))
            _quantile_ribbon(ax, flat)
            ax.set_title(f"{name} value dist")
            ax.set_xlabel('Epoch'); ax.set_ylabel('Value')
            _save(fig, f"{name}_dist.png")
            norms = np.sqrt((flat**2).sum(axis=1))
            fig, ax = plt.subplots(figsize=(5,3))
            ax.plot(norms); ax.set_title(f"{name} Fro norm")
            ax.set_xlabel('Epoch')
            _save(fig, f"{name}_norm.png")

            last = _to_np(series[-1])
            if last.ndim == 3:
                fig, ax = plt.subplots(figsize=(4,4))
                sns.heatmap(last[0], ax=ax, cmap='viridis')
                ax.set_title(f"{name} last[0]")
                _save(fig, f"{name}_last_slice.png")

        # Specialized diagnostics
        # HMM specifics
        if model_type == 'hmm':
            init_series = hist.get('initial_logits')
            trans_series = hist.get('transition_logits')
            mean_series = hist.get('emission_mean')
            logvar_series = hist.get('emission_logvar')
            chol_series = hist.get('emission_cholesky_raw')

            if init_series:
                probs = [torch.softmax(torch.as_tensor(s), -1).cpu().numpy() for s in init_series]
                ent = [- (p * np.log(np.clip(p,1e-12,1))).sum() for p in probs]
                fig, ax = plt.subplots(figsize=(5,3))
                ax.plot(ent); ax.set_title('Initial entropy'); ax.set_xlabel('Epoch'); ax.set_ylabel('H')
                _save(fig, 'initial_entropy.png')

            if trans_series:
                Tmats = [torch.softmax(torch.as_tensor(s), -1).cpu().numpy() for s in trans_series]
                row_ent = np.stack([(-tm * np.log(np.clip(tm,1e-12,1))).sum(axis=1) for tm in Tmats],0)
                diag_p = np.stack([np.diag(tm) for tm in Tmats],0)
                fig, ax = plt.subplots(1,2, figsize=(10,3))
                _quantile_ribbon(ax[0], row_ent, label='row entropy'); ax[0].legend(frameon=False)
                ax[0].set_title('Transition row entropy'); ax[0].set_xlabel('Epoch'); ax[0].set_ylabel('H')
                ax[1].plot(diag_p); ax[1].set_title('Transition diag probs'); ax[1].set_xlabel('Epoch'); ax[1].set_ylabel('P(stay)')
                _save(fig, 'transition_entropy_diag.png')

            if mean_series is not None:
                means = np.stack([_to_np(s) for s in mean_series],0) # (E,S,D)
                norms = np.linalg.norm(means, axis=2)
                fig, ax = plt.subplots(figsize=(7,3.2))
                ax.plot(norms); ax.set_title('Emission mean norms per state'); ax.set_xlabel('Epoch'); ax.set_ylabel('||mu_s||')
                _save(fig, 'emission_mean_norms.png')
                E = means.shape[0]
                for idx in _first_mid_last(E):
                    fig, hx = plt.subplots(figsize=(5,4))
                    sns.heatmap(means[idx], cmap='coolwarm', ax=hx)
                    hx.set_title(f'Means e{idx} (S x D)')
                    _save(fig, f'emission_mean_e{idx}.png')

            if logvar_series is not None:
                logv = np.stack([_to_np(s) for s in logvar_series],0)
                var = np.exp(logv)
                flat = var.reshape(var.shape[0], -1)
                fig, ax = plt.subplots(figsize=(6,3.2))
                _quantile_ribbon(ax, flat, label='variance'); ax.set_yscale('log'); ax.legend(frameon=False)
                ax.set_title('Emission variance quantiles'); ax.set_xlabel('Epoch'); ax.set_ylabel('Var')
                _save(fig, 'emission_variance_quantiles.png')

            if chol_series is not None:
                logdets = []
                for raw in chol_series:
                    r = torch.as_tensor(raw)
                    tril = torch.tril(r)
                    diag = torch.diagonal(tril, dim1=-2, dim2=-1)
                    diag = torch.nn.functional.softplus(diag) + 1e-5
                    logdet = 2 * torch.log(diag).sum(-1)
                    logdets.append(logdet.cpu().numpy())
                logdets = np.stack(logdets,0)
                fig, ax = plt.subplots(figsize=(7,3))
                ax.plot(logdets); ax.set_title('Log-det covariance per state'); ax.set_xlabel('Epoch'); ax.set_ylabel('log|Σ|')
                _save(fig, 'emission_logdet.png')

        # MARHMM specifics
        if model_type == 'marhmm':
            coeff_series = hist.get('coeffs')
            bias_series = hist.get('bias')
            logvar_series = hist.get('log_var')
            trans_series = hist.get('transition_logits')

            if coeff_series is not None and coeff_series:
                coeffs = [ _to_np(s) for s in coeff_series ]
                S,D,DL = coeffs[0].shape
                L = DL // D if D>0 else 0
                state_norms = []
                lag_norms = []
                for c in coeffs:
                    resh = c.reshape(S,D,L,D)
                    state_norms.append(np.sqrt((resh**2).sum(axis=(1,2,3))))
                    lag_norms.append(np.sqrt((resh**2).sum(axis=(0,1,3))))
                state_norms = np.stack(state_norms,0)
                lag_norms = np.stack(lag_norms,0)
                fig, ax = plt.subplots(figsize=(7,3))
                ax.plot(state_norms); ax.set_title('AR coeff norms per state'); ax.set_xlabel('Epoch'); ax.set_ylabel('Norm')
                _save(fig, 'ar_state_norms.png')
                fig, ax = plt.subplots(figsize=(6,3))
                ax.plot(lag_norms); ax.set_title('AR coeff norms per lag'); ax.set_xlabel('Epoch'); ax.set_ylabel('Norm')
                _save(fig, 'ar_lag_norms.png')
                last = coeffs[-1].reshape(S,D,L,D)
                mean_abs = np.mean(np.abs(last), axis=(1,3))  # (S,L)
                fig, ax = plt.subplots(figsize=(5,4))
                sns.heatmap(mean_abs, cmap='magma', ax=ax)
                ax.set_title('Mean |coeff| last (state x lag)')
                ax.set_xlabel('Lag'); ax.set_ylabel('State')
                _save(fig, 'ar_mean_abs_last.png')

            if bias_series is not None and bias_series:
                biases = np.stack([_to_np(s) for s in bias_series],0) # (E,S,D)
                norms = np.linalg.norm(biases, axis=2)
                fig, ax = plt.subplots(figsize=(6,3))
                ax.plot(norms); ax.set_title('Bias norms per state'); ax.set_xlabel('Epoch'); ax.set_ylabel('||b_s||')
                _save(fig, 'bias_norms.png')

            if logvar_series is not None and logvar_series:
                lv = np.stack([_to_np(s) for s in logvar_series],0)
                var = np.exp(lv)
                flat = var.reshape(var.shape[0], -1)
                fig, ax = plt.subplots(figsize=(6,3))
                _quantile_ribbon(ax, flat, label='variance'); ax.set_yscale('log'); ax.legend(frameon=False)
                ax.set_title('Emission variance quantiles'); ax.set_xlabel('Epoch'); ax.set_ylabel('Var')
                _save(fig, 'emission_variances.png')

            if trans_series is not None and trans_series:
                Tmats = [torch.softmax(torch.as_tensor(s), -1).cpu().numpy() for s in trans_series]
                diag_p = np.stack([np.diag(tm) for tm in Tmats],0)
                fig, ax = plt.subplots(figsize=(6,3))
                ax.plot(diag_p); ax.set_title('Transition diag probabilities'); ax.set_xlabel('Epoch'); ax.set_ylabel('P(stay)')
                _save(fig, 'transition_diag_probs.png')

        # Optional: interactive sliders for small params (kept lightweight)
        for name, series in hist.items():
            try:
                sample = _to_np(series[0])
                if sample.ndim <= 2 and sample.size <= 4000:  # avoid huge HTML
                    frames = []
                    for i, s in enumerate(series):
                        arr2 = _to_np(s)
                        if arr2.ndim == 1:
                            frames.append(go.Frame(data=[go.Scatter(y=arr2, mode='lines')], name=str(i)))
                        elif arr2.ndim == 2:
                            frames.append(go.Frame(data=[go.Heatmap(z=arr2)], name=str(i)))
                        else:
                            continue
                    if frames:
                        if sample.ndim == 1:
                            fig = go.Figure(data=[go.Scatter(y=_to_np(series[0]), mode='lines')], frames=frames)
                        else:
                            fig = go.Figure(data=[go.Heatmap(z=_to_np(series[0]))], frames=frames)
                        fig.update_layout(title=f"{name} (epoch slider)", updatemenus=[{'type':'buttons','buttons':[{'label':'Play','method':'animate','args':[None,{'frame':{'duration':120,'redraw':True},'fromcurrent':True}]},{'label':'Pause','method':'animate','args':[[None],{'frame':{'duration':0,'redraw':False}}]}]}], sliders=[{'steps':[{'args':[[str(i)],{'frame':{'duration':0,'redraw':True},'mode':'immediate'}],'label':str(i),'method':'animate'} for i in range(len(frames))], 'currentvalue':{'prefix':'Epoch: '}}])
                        fig.write_html(str(base / f"{name}_slider.html"))
            except Exception as e:
                print(f"Interactive slider for {name} failed: {e}")

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

    