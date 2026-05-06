from typing import Any, Optional, Sequence
from pathlib import Path

from src.models.base_model import BaseModel
from src.models.cvae_mar_hmm import CVAEMARHMM
from torch import Tensor
import numpy as np
import seaborn as sns
import matplotlib
# Force a non-interactive backend to avoid Tkinter dependency/issues on Windows or headless runs
try:
    if str(matplotlib.get_backend()).lower() != 'agg':
        matplotlib.use('Agg')
except Exception:
    # If backend is already set or unavailable, ignore and proceed
    pass
import matplotlib.pyplot as plt
import torch
import plotly.graph_objects as go
from src.config.config import GlobalConfig
from src.data.data_loader import DataLoader
from src.data.data_loader_collection import DataLoaderCollection
from src.helpers.align_labels import align_labels_hungarian
from src.orchestrator.train_details import TrainDetails
from src.validation.validator import Validator
from sklearn.metrics import confusion_matrix
from itertools import combinations

class Visualizer:
    def __init__(self,
                 data_loader: DataLoaderCollection,
                 config: GlobalConfig,
                 model: BaseModel,
                 validator: Validator):
        self.data_loader = data_loader
        self.global_config = config
        self.model = model
        self.config = self.global_config.visualizer
        self.validator = validator

    def _subject_ids_for_export(self, ids: torch.Tensor) -> np.ndarray:
        if ids.dim() >= 3 and ids.shape[-1] >= 2:
            ids = ids[..., 0]
        return ids.flatten().cpu().numpy()
    
    ####### GENERAL METHODS #######
    
    def visualize_data(self):
        path = Path(self.global_config.results_dir) / self.global_config.run_name / "plots"
        path.mkdir(parents=True, exist_ok=True)
        try:
            dv = self.validator.get_data_validations()
        except Exception as e:
            print('There was no data validations to visualize.')
            return
        if self.global_config.visualizer.state_distinctness and self.global_config.validator.state_distinctness:
            self.__plot_state_distinctness(path=path)
        if self.global_config.visualizer.summary_statistics and self.global_config.validator.summary_statistics:
            self.__plot_summary_statistics(path=path)
            self.__plot_feature_statistics(path=path)
            self.__plot_feature_correlations(path=path)
        self.__plot_feature_separability(path=path)
    
    
    def visualize(self, train_details: TrainDetails):
        path = train_details.get_path() / "plots"
        path.mkdir(parents=True, exist_ok=True)
        x, y, subject_ids = self.data_loader.get_all_data()
        x_non_norm = self.data_loader.get_non_normalized_data()
        if self.global_config.model.type == 'marhmm':
            x = x[:, self.global_config.model.params['lags'][-1]:]
            y = y[:, self.global_config.model.params['lags'][-1]:]
        if self.global_config.model.type == 'cvae_marhmm':
            x = self.model.get_latent_representation(x, subject_ids)
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
        if self.config.learning_rate:
            self.__plot_learning_rate(train_details)
        y_hat = train_details.get_trained_predictions()
        subject_ids = self._subject_ids_for_export(subject_ids)
        np.savez(path / "results.npz", y_hat=y_hat, y_true=y, x_latent=x, x=x_non_norm, sub_ids=subject_ids)
    
    
    def visualize_cvae_gmm(self, y_hat, y_true, x_latent: torch.Tensor, x: torch.Tensor, nmi, likelihood, sub_ids):
        path = Path(self.global_config.results_dir) / self.global_config.run_name / "plots"
        path.mkdir(parents=True, exist_ok=True)
        y_hat = y_hat.flatten().cpu().numpy()
        y_true = y_true.flatten().cpu().numpy()
        x_latent = x_latent.detach().cpu().numpy()
        sub_ids = self._subject_ids_for_export(sub_ids)
        self.__plot_pca_tripanel(None, x=x_latent, y=y_true, overwrite_path=path, pred_y=y_hat)
        # write a txt file with nmi and likelihood
        with open(path / "metrics.txt", "w") as f:
            f.write(f"NMI: {nmi}\n")
            f.write(f"Likelihood: {likelihood}\n")
        # write y_hat, y_true, x_latent to npz
        np.savez(path / "results.npz", y_hat=y_hat, y_true=y_true, x_latent=x_latent, x=x, sub_ids=sub_ids)
        
    
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
    
    def visualize_cvae(self, model: CVAEMARHMM, train_details):
        path = Path(self.global_config.results_dir) / self.global_config.run_name / "plots"
        path.mkdir(parents=True, exist_ok=True)
        x, y, sub_ids = self.data_loader.get_all_data()
        y = y.flatten().cpu().numpy()
        model.prepare_for_inference()
        with torch.no_grad():
            x_latent = model.get_latent_representation(x, sub_ids).cpu().numpy()
        self.__plot_feature_statistics(path=path)
        self.__plot_state_distinctness(path=path)
        self.__plot_pca_tripanel(train_details, x=x_latent, y=y, overwrite_path=path)
    
    ####### HELPER METHODS #######
    
    def __plot_feature_statistics(self, path: Path):
        """Visualise per-state feature statistics supplied by the validator."""
        dv = self.validator.get_data_validations()
        raw_stats = dv.get("feature_statistics", {})
        if not raw_stats:
            return

        def _to_array(value: Any) -> np.ndarray:
            if value is None:
                return np.asarray([], dtype=float)
            if hasattr(value, "detach"):
                try:
                    return value.detach().cpu().numpy().reshape(-1)
                except Exception:
                    return np.asarray(value).reshape(-1)
            arr = np.asarray(value)
            if arr.size == 0:
                return np.asarray([], dtype=float)
            return arr.reshape(-1)

        # Normalise structure so we always operate on metric -> state -> feature array
        metrics: dict[str, dict[str, Any]]
        if isinstance(raw_stats, dict) and raw_stats and all(isinstance(v, dict) for v in raw_stats.values()):
            metrics = raw_stats  # already metric keyed
        else:
            metrics = {"amplitude": raw_stats}

        std_lookup: dict[str, dict[str, Any]] = {
            key[:-4]: value for key, value in metrics.items()
            if key.endswith('_std') and isinstance(value, dict)
        }

        def key_fn(key: str):
            try:
                return int(key)
            except Exception:
                return key

        # Collect the union of states across all metrics to ensure consistent ordering
        state_keys: set[str] = set()
        for metric_values in metrics.values():
            state_keys.update(metric_values.keys())
        states = sorted(state_keys, key=key_fn)
        if not states:
            return

        state_names = self.data_loader.get_state_names() or []
        display_labels: dict[str, str] = {}
        for s in states:
            label = s
            try:
                if str(s).isdigit():
                    idx = int(s)
                    if 0 <= idx < len(state_names):
                        label = state_names[idx]
            except Exception:
                pass
            display_labels[s] = label

        sns.set_style('whitegrid')
        palette_name = 'tab10' if len(states) <= 10 else 'tab20' if len(states) <= 20 else 'hsv'
        try:
            base_colors = sns.color_palette(palette_name, len(states))
        except Exception:
            base_colors = sns.color_palette('tab10', len(states))

        metric_store: dict[str, dict[str, np.ndarray]] = {}
        std_store: dict[str, dict[str, np.ndarray]] = {}
        global_max_features = 0

        for metric_key in sorted(metrics.keys()):
            if metric_key.endswith('_std'):
                continue

            metric_values = metrics[metric_key]
            metric_title = metric_key.replace('_', ' ').title()
            std_values = std_lookup.get(metric_key)

            # Gather arrays per state and track maximum feature count
            arrays: dict[str, np.ndarray] = {}
            std_arrays: dict[str, np.ndarray] | None = {} if std_values is not None else None
            max_features = 0
            for s in states:
                arr = _to_array(metric_values.get(s))
                arrays[s] = arr
                max_features = max(max_features, arr.size)
                if std_arrays is not None:
                    std_arr = _to_array(std_values.get(s)) if std_values is not None else np.asarray([], dtype=float)
                    std_arrays[s] = std_arr
                    max_features = max(max_features, std_arr.size)

            if std_arrays is not None:
                has_valid_std = any(std_arr.size and np.any(np.isfinite(std_arr)) for std_arr in std_arrays.values())
                if not has_valid_std:
                    std_arrays = None

            if max_features == 0:
                continue
            
            metric_store[metric_key] = arrays
            if std_arrays is not None:
                std_store[metric_key] = std_arrays
            global_max_features = max(global_max_features, max_features)

            feature_axis = np.arange(1, max_features + 1)
            # feature_labels = [f"Feature {i}" for i in feature_axis]
            feature_labels = self.data_loader.get_feature_names()
            if not len(feature_labels) == max_features:
                feature_labels = [f"Feature {i}" for i in feature_axis]

            # Line plot with optional std shading
            fig, ax = plt.subplots(figsize=(12, 6))
            for idx, s in enumerate(states):
                arr = arrays[s]
                padded = np.full(max_features, np.nan)
                if arr.size:
                    padded[:arr.size] = arr
                color = base_colors[idx % len(base_colors)]

                if std_arrays is not None:
                    std_arr = std_arrays.get(s, np.asarray([], dtype=float))
                    std_padded = np.full(max_features, np.nan)
                    if std_arr.size:
                        std_padded[:std_arr.size] = std_arr
                    valid_mask = np.isfinite(std_padded) & np.isfinite(padded)
                    if valid_mask.any():
                        lower = padded.copy()
                        upper = padded.copy()
                        lower[valid_mask] = padded[valid_mask] - std_padded[valid_mask]
                        upper[valid_mask] = padded[valid_mask] + std_padded[valid_mask]
                        ax.fill_between(feature_axis, lower, upper, where=valid_mask,
                                        color=color, alpha=0.18, linewidth=0)

                ax.plot(feature_axis, padded, label=display_labels[s],
                        color=color, linewidth=1.8, alpha=0.9)

            ax.set_xlabel('Feature', fontsize=12)
            ax.set_ylabel(metric_title, fontsize=12)
            ax.set_title(f'{metric_title} Per State Across Features', fontsize=14)
            ax.set_xticks(feature_axis)
            ax.set_xticklabels(feature_labels, rotation=45, ha='right')
            ax.tick_params(axis='both', which='major', labelsize=10)
            ax.grid(True, which='both', linestyle='--', linewidth=0.4, alpha=0.6)
            ax.legend(loc='upper right', fontsize='small', frameon=False)
            fig.tight_layout()
            fig.savefig(path / f'feature_{metric_key.lower()}_per_state.png', dpi=300, bbox_inches='tight')
            plt.close(fig)

            # Heatmap: states x features for this metric
            mat = np.full((len(states), max_features), np.nan)
            for row_idx, s in enumerate(states):
                arr = arrays[s]
                if arr.size:
                    mat[row_idx, :min(arr.size, max_features)] = arr[:max_features]

            fig2, ax2 = plt.subplots(figsize=(12, max(2.5, len(states) * 0.6)))
            sns.heatmap(mat, ax=ax2, cmap='viridis', cbar_kws={'label': metric_title},
                        xticklabels=feature_labels, yticklabels=[display_labels[s] for s in states])
            ax2.set_xlabel('Feature')
            ax2.set_title(f'{metric_title} Heatmap (State × Feature)')
            # Make x labels much smaller and keep rotation for readability
            plt.setp(ax2.get_xticklabels(), fontsize=6, rotation=45, ha='right')
            # Make y labels horizontal
            plt.setp(ax2.get_yticklabels(), fontsize=9, rotation=0, ha='right')
            fig2.tight_layout()
            fig2.savefig(path / f'feature_{metric_key.lower()}_heatmap.png', dpi=250)
            plt.close(fig2)

        metrics_to_plot = list(metric_store.keys())
        if not metrics_to_plot or global_max_features == 0:
            return

        metric_palette = sns.color_palette('deep', len(metrics_to_plot))

        for feat_idx in range(global_max_features):
            feature_label = f'Feature {feat_idx + 1}'
            x_positions = np.arange(len(states))

            values_by_metric: list[np.ndarray] = []
            errors_by_metric: list[np.ndarray | None] = []
            for metric_key in metrics_to_plot:
                arrays = metric_store.get(metric_key, {})
                std_arrays = std_store.get(metric_key)

                metric_values = []
                metric_errors = []
                for s in states:
                    arr = arrays.get(s, np.asarray([], dtype=float))
                    val = float(arr[feat_idx]) if arr.size > feat_idx else np.nan
                    metric_values.append(val)
                    if std_arrays is not None:
                        std_arr = std_arrays.get(s, np.asarray([], dtype=float))
                        err_val = float(std_arr[feat_idx]) if std_arr.size > feat_idx else np.nan
                    else:
                        err_val = np.nan
                    metric_errors.append(err_val)

                values_by_metric.append(np.asarray(metric_values, dtype=float))
                if std_arrays is not None and not np.all(np.isnan(metric_errors)):
                    errors_by_metric.append(np.asarray(metric_errors, dtype=float))
                else:
                    errors_by_metric.append(None)

            if all(np.all(np.isnan(vals)) for vals in values_by_metric):
                continue

            width = 0.8 / max(1, len(metrics_to_plot))
            fig_feat, ax_feat = plt.subplots(figsize=(max(7, len(states) * 1.1), 4.5))

            for metric_idx, metric_key in enumerate(metrics_to_plot):
                offset = (metric_idx - (len(metrics_to_plot) - 1) / 2) * width
                positions = x_positions + offset
                values = values_by_metric[metric_idx]
                errors = errors_by_metric[metric_idx]

                if np.all(np.isnan(values)):
                    continue

                bar_kwargs: dict[str, Any] = {
                    'x': positions,
                    'height': values,
                    'width': width * 0.9,
                    'color': metric_palette[metric_idx % len(metric_palette)],
                    'alpha': 0.9,
                    'label': metric_key.replace('_', ' ').title(),
                }

                if errors is not None:
                    bar_kwargs['yerr'] = errors
                    bar_kwargs['capsize'] = 4
                    bar_kwargs['error_kw'] = {'elinewidth': 1, 'alpha': 0.7}

                ax_feat.bar(**bar_kwargs)

            ax_feat.set_xticks(x_positions)
            ax_feat.set_xticklabels([display_labels[s] for s in states], rotation=30, ha='right')
            ax_feat.set_ylabel('Value')
            ax_feat.set_xlabel('State')
            ax_feat.set_title(f'{feature_label} – Metrics by State')
            ax_feat.legend(loc='upper right', fontsize='small', frameon=False)
            ax_feat.grid(True, axis='y', linestyle='--', linewidth=0.4, alpha=0.6)

            fig_feat.tight_layout()
            out_name = f'feature_summary_{feat_idx + 1:02d}.png'
            fig_feat.savefig(path / out_name, dpi=250)
            plt.close(fig_feat)

    def __plot_feature_correlations(self, path: Path) -> None:
        """Plot feature-feature correlation heatmaps (overall and per-state) from validator data."""
        dv = self.validator.get_data_validations()
        corr = dv.get("feature_correlations", {})
        if not corr:
            return

        overall = np.asarray(corr.get("overall", []), dtype=float)
        per_state: dict[str, Any] = corr.get("per_state", {}) or {}
        feature_names = corr.get("feature_names") or self.data_loader.get_feature_names() or []

        if overall.size:
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(overall, ax=ax, cmap='coolwarm', center=0.0, vmin=-1.0, vmax=1.0,
                        xticklabels=feature_names if len(feature_names) == overall.shape[1] else True,
                        yticklabels=feature_names if len(feature_names) == overall.shape[0] else True,
                        square=True, cbar_kws={'label': 'Pearson r'})
            ax.set_title('Feature Correlation (Overall)')
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
            plt.setp(ax.get_yticklabels(), rotation=0, ha='right', fontsize=8)
            fig.tight_layout()
            fig.savefig(path / 'feature_correlation_overall.png', dpi=250)
            plt.close(fig)

        # Per-state heatmaps
        if per_state:
            state_names = self.data_loader.get_state_names() or []
            for k, mat in per_state.items():
                mat_np = np.asarray(mat, dtype=float)
                if mat_np.size == 0:
                    continue
                try:
                    idx = int(k)
                except Exception:
                    idx = None
                label = state_names[idx] if (idx is not None and 0 <= idx < len(state_names)) else str(k)
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(mat_np, ax=ax, cmap='coolwarm', center=0.0, vmin=-1.0, vmax=1.0,
                            xticklabels=feature_names if len(feature_names) == mat_np.shape[1] else True,
                            yticklabels=feature_names if len(feature_names) == mat_np.shape[0] else True,
                            square=True, cbar_kws={'label': 'Pearson r'})
                ax.set_title(f'Feature Correlation – State {label}')
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
                plt.setp(ax.get_yticklabels(), rotation=0, ha='right', fontsize=8)
                fig.tight_layout()
                safe_label = str(label).replace(' ', '_')
                fig.savefig(path / f'feature_correlation_state_{safe_label}.png', dpi=230)
                plt.close(fig)

    def __plot_feature_separability(self, path: Path) -> None:
        dv = self.validator.get_data_validations()
        separability = dv.get("feature_separability", {})
        if not separability:
            return

        scores = np.asarray(separability.get("scores", []), dtype=float)
        if scores.size == 0:
            return

        feature_names: Optional[Sequence[str]] = separability.get("feature_names")
        if not feature_names or len(feature_names) != scores.size:
            fallback_names = self.data_loader.get_feature_names() or []
            if len(fallback_names) != scores.size:
                feature_names = [f"Feature {i + 1}" for i in range(scores.size)]
            else:
                feature_names = [str(name) for name in fallback_names]
        else:
            feature_names = [str(name) for name in feature_names]

        order = np.argsort(scores)[::-1]
        sorted_scores = scores[order]
        sorted_names = [feature_names[idx] for idx in order]

        sns.set_style('whitegrid')
        fig_height = max(4.0, 0.35 * len(sorted_scores))
        fig, ax = plt.subplots(figsize=(12, fig_height))

        y_pos = np.arange(len(sorted_scores))
        palette = sns.cubehelix_palette(len(sorted_scores), start=0.6, rot=-0.75)
        bars = ax.barh(y_pos, sorted_scores, color=palette)
        ax.invert_yaxis()
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_names)
        ax.set_xlabel('Fisher score (higher is better)')
        ax.set_title('Feature separability ranking')

        max_score = float(np.nanmax(sorted_scores)) if np.isfinite(sorted_scores).any() else 0.0
        offset = max(max_score * 0.015, 0.01)
        # Annotate each bar with its score so analysts can compare relative separability at a glance.
        for idx, bar in enumerate(bars):
            width = float(bar.get_width())
            ax.text(width + offset, bar.get_y() + bar.get_height() / 2.0,
                    f"{sorted_scores[idx]:.3f}", va='center', ha='left', fontsize=8, color='#333333')

        fig.tight_layout()
        fig.savefig(path / 'feature_separability_ranking.png', dpi=220, bbox_inches='tight')
        plt.close(fig)
    
    def __plot_summary_statistics(self, path: Path):
        # load from validator
        dv = self.validator.get_data_validations()
        input_mean = dv.get("input_mean", None) # float
        input_std = dv.get("input_std", None) # float
        target_classes = dv.get("target_classes", None) # list[int]
        target_counts = dv.get("target_counts", None) # dict[int, float]
        target_means = dv.get("target_means", None) # dict[int, float]
        target_stds = dv.get("target_stds", None) # dict[int, float]
        state_names = self.data_loader.get_state_names()

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
        state_names = self.data_loader.get_state_names()

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
        state_names = self.data_loader.get_state_names()
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
        labels = self.data_loader.get_state_names()
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
    
    def __plot_metrics_over_epochs(self, train_details: TrainDetails):
        """Create a beautiful line plot showing NMI and Accuracy over training epochs using seaborn styling."""
        if not train_details.validations:
            return  # No validation data available
        
        out_dir = train_details.get_path() / "plots"
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract epochs and metrics
        epochs = sorted(train_details.validations.keys())
        nmi_values: list[float | None] = []
        acc_values: list[float | None] = []

        for epoch in epochs:
            validation_data = train_details.validations[epoch]
            nmi_val = validation_data.get('nmi')
            acc_val = validation_data.get('accuracy')
            nmi_values.append(nmi_val if isinstance(nmi_val, (int, float)) else None)
            acc_values.append(acc_val if isinstance(acc_val, (int, float)) else None)

        # Losses (may have different epoch coverage)
        loss_epochs = sorted(train_details.losses.keys()) if getattr(train_details, 'losses', None) else []
        loss_values: list[float] = [train_details.losses[e] for e in loss_epochs]
        
        # Set seaborn style for beautiful plots
        sns.set_style("whitegrid")
        sns.set_palette("husl")
        
        # Create matplotlib figure with seaborn styling
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Convert to 1-based epochs for display
        epoch_display = [e + 1 for e in epochs]
        loss_epoch_display = [e + 1 for e in loss_epochs]

        # Union for axis limits (fall back to metrics if no losses)
        all_epoch_display = epoch_display if not loss_epoch_display else sorted(set(epoch_display) | set(loss_epoch_display))
        
        # Define beautiful colors
        nmi_color = '#2E86AB'  # Beautiful blue
        acc_color = '#A23B72'  # Beautiful magenta/purple
        loss_color = '#555555'  # Neutral gray for loss
        
        # Plot NMI as a plain line; annotate earliest max
        if any(v is not None for v in nmi_values):
            valid_nmi = [(x, y) for x, y in zip(epoch_display, nmi_values) if y is not None]
            if valid_nmi:
                x_nmi, y_nmi = zip(*valid_nmi)
                ax.plot(x_nmi, y_nmi, '-', color=nmi_color, linewidth=2.5,
                        label='NMI (Normalized Mutual Information)', alpha=0.9)
                max_nmi = max(y_nmi)
                # earliest occurrence
                for idx, v in enumerate(y_nmi):
                    if v == max_nmi:
                        max_nmi_idx = idx
                        break
                max_nmi_epoch = x_nmi[max_nmi_idx]
                ax.annotate(f"Peak NMI {max_nmi:.3f} (E{max_nmi_epoch})",
                            xy=(max_nmi_epoch, max_nmi),
                            xytext=(10, 6), textcoords='offset points',
                            fontsize=9, fontweight='bold', color=nmi_color,
                            bbox=dict(boxstyle='round,pad=0.25', fc='white', ec=nmi_color, lw=0.8, alpha=0.85),
                            arrowprops=dict(arrowstyle='->', color=nmi_color, lw=0.8))
        
        # Plot Accuracy (if still present in validations) as a plain line; earliest max
        if any(v is not None for v in acc_values):
            valid_acc = [(x, y) for x, y in zip(epoch_display, acc_values) if y is not None]
            if valid_acc:
                x_acc, y_acc = zip(*valid_acc)
                ax.plot(x_acc, y_acc, '-', color=acc_color, linewidth=2.5,
                        label='Accuracy', alpha=0.75)
                max_acc = max(y_acc)
                for idx, v in enumerate(y_acc):
                    if v == max_acc:
                        max_acc_idx = idx
                        break
                max_acc_epoch = x_acc[max_acc_idx]
                ax.annotate(f"Peak Acc {max_acc:.3f} (E{max_acc_epoch})",
                            xy=(max_acc_epoch, max_acc),
                            xytext=(8, -18), textcoords='offset points',
                            fontsize=8, fontweight='bold', color=acc_color,
                            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=acc_color, lw=0.6, alpha=0.85),
                            arrowprops=dict(arrowstyle='->', color=acc_color, lw=0.6))

        # Secondary axis for Loss
        ax2 = None
        if loss_values:
            ax2 = ax.twinx()
            ax2.plot(loss_epoch_display, loss_values, '-', color=loss_color, linewidth=2.0,
                     label='Loss', alpha=0.9)
            ax2.set_ylabel('Loss', fontsize=14, fontweight='medium', color=loss_color)
            ax2.tick_params(axis='y', labelcolor=loss_color)
            # Annotate earliest min loss
            min_loss = min(loss_values)
            for idx, v in enumerate(loss_values):
                if v == min_loss:
                    min_loss_idx = idx
                    break
            min_loss_epoch = loss_epoch_display[min_loss_idx]
            ax2.annotate(f"Min Loss {min_loss:.4g} (E{min_loss_epoch})",
                         xy=(min_loss_epoch, min_loss),
                         xytext=(10, -20), textcoords='offset points',
                         fontsize=9, fontweight='bold', color=loss_color,
                         bbox=dict(boxstyle='round,pad=0.25', fc='white', ec=loss_color, lw=0.8, alpha=0.85),
                         arrowprops=dict(arrowstyle='->', color=loss_color, lw=0.7))
        
        # Enhanced customization
        ax.set_xlabel('Training Epoch', fontsize=14, fontweight='medium')
        ax.set_ylabel('Metric Value', fontsize=14, fontweight='medium')
        ax.set_title(f'Training Performance Metrics - Run {train_details.run_number}', 
                    fontsize=16, fontweight='bold', pad=20)
        
        # Set limits and ticks
        ax.set_ylim(-0.05, 1.05)  # Slightly expanded for visual breathing room for metrics
        ax.set_xlim(min(all_epoch_display) - 0.5, max(all_epoch_display) + 0.5)
        
        # Enhanced grid
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.8)
        ax.set_axisbelow(True)  # Grid behind the lines
        
        # Beautiful legend
        # Combine legends from both axes if loss present
        if ax2 is not None:
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            legend = ax.legend(lines1 + lines2, labels1 + labels2, fontsize=12, frameon=True, fancybox=True, shadow=True,
                               loc='best', borderpad=1, columnspacing=1.5)
        else:
            legend = ax.legend(fontsize=12, frameon=True, fancybox=True, shadow=True, 
                               loc='best', borderpad=1, columnspacing=1.5)
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_alpha(0.9)
        legend.get_frame().set_edgecolor('lightgray')
        
        # Add subtle background
        ax.set_facecolor('#FAFAFA')
        if ax2 is not None:
            ax2.set_facecolor('#FAFAFA')
        
        # Enhance tick parameters
        ax.tick_params(axis='both', which='major', labelsize=11, 
                      colors='#333333', width=1, length=6)
        ax.tick_params(axis='both', which='minor', width=0.5, length=3)
        if ax2 is not None:
            ax2.tick_params(axis='y', which='major', labelsize=11, width=1, length=6)
        
        # Add minor ticks for better granularity
        ax.minorticks_on()
        
        # Spines styling
        for spine in ax.spines.values():
            spine.set_color('#CCCCCC')
            spine.set_linewidth(1)
        if ax2 is not None:
            for spine in ax2.spines.values():
                spine.set_color('#BBBBBB')
                spine.set_linewidth(1)
        
        # Save as PNG with high quality
        png_path = out_dir / 'metrics_over_epochs.png'
        fig.tight_layout(pad=2.0)
        fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close(fig)
        
        # Reset seaborn style to not affect other plots
        sns.reset_defaults()
        
        if self.global_config.verbose:
            if any(v is not None for v in nmi_values):
                print(f"Max NMI at epoch {max_nmi_epoch}: {max_nmi:.4f}")
            if any(v is not None for v in acc_values):
                print(f"Max Accuracy at epoch {max_acc_epoch}: {max_acc:.4f}")
            if loss_values:
                print(f"Min Loss at epoch {min_loss_epoch}: {min_loss:.6g}")

    def __plot_pca_tripanel(self, train_details: TrainDetails, x: Tensor, y: Tensor, overwrite_path: str = None, pred_y: np.ndarray = None):
        """Save tri-panel PCA plots comparing HMM-init, HMM-trained, and True labels."""
        
        if len(y.shape) == 3:
            y = y[:, :, 0]
        
        if train_details is None and pred_y is None:
            return
        
        if pred_y is not None:
            init_arr = pred_y
            trained_arr = pred_y
        else:
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
            print(f"Lengths: y={len(y)}, init={len(init_arr)}, trained={len(trained_arr)}, X_rows={X.shape[0]}")
            raise ValueError("Label lengths must match number of rows in x after flattening")
        
        ## sample 2500 points from each class
        sampled_indices = []
        for cls in np.unique(y):
            cls_indices = np.where(y == cls)[0]
            if len(cls_indices) > 2000:
                sampled = np.random.choice(cls_indices, size=2000, replace=False)
            else:
                sampled = cls_indices
            sampled_indices.extend(sampled)
        sampled_indices = np.array(sampled_indices)
        X = X[sampled_indices]
        init_arr = init_arr[sampled_indices]
        trained_arr = trained_arr[sampled_indices]
        y = y[sampled_indices]
        
        # PCA via SVD (up to 4 comps)
        K = min(4, max(2, X.shape[1]))
        U, S, _ = np.linalg.svd(X - X.mean(0, keepdims=True), full_matrices=False)
        proj = U[:, :K] * S[:K]

        # Colors and titles
        palette = np.array(["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"])
        def colors(a):
            u = np.unique(a)
            lut = {v: palette[i % len(palette)] for i, v in enumerate(u)}
            return np.array([lut[v] for v in a])

        cols = [colors(init_arr), colors(trained_arr), colors(y)]
        titles = ["HMM init", "HMM trained", "True"]
        arrays = [init_arr, trained_arr, y]

        # human-readable state names (if available)
        state_names = self.data_loader.get_state_names()

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
            out_path = Path(overwrite_path) / f"hmm_tripanel_pc{a+1}_pc{b+1}.png" if overwrite_path else train_details.get_path() / "plots" / f"hmm_tripanel_pc{a+1}_pc{b+1}.png"
            fig.savefig(out_path.as_posix(), dpi=160)
            plt.close(fig)
            saved.append(out_path.as_posix())   
        return saved
    
    def __plot_learning_rate(self, train_details: TrainDetails):
        """Visualize the learning rate over epochs."""
        vd = train_details.get_validations()
        lr_history = []
        for epoch in sorted(vd.keys()):
            lr = vd[epoch].get('learning_rate', None)
            if lr is not None:
                lr_history.append(lr)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(lr_history, label='Learning Rate', color='tab:blue')
        ax.set_title('Learning Rate Over Epochs')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.legend()
        out_path = train_details.get_path() / "plots" / "learning_rate.png"
        fig.savefig(out_path.as_posix(), dpi=160)
        plt.close(fig)

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
        import traceback
        for name, series in hist.items():
            try:
                # Basic sanity: need at least one sample and not too large
                if not series:
                    continue
                sample = _to_np(series[0])
                if sample is None:
                    continue
                if sample.ndim > 2 or sample.size > 4000:
                    # Skip very large or high-rank tensors
                    continue

                # Ensure all frames are consistent (same ndim and shape)
                target_ndim = sample.ndim
                target_shape = sample.shape
                frames = []
                for i, s in enumerate(series):
                    arr2 = _to_np(s)
                    if arr2 is None:
                        continue
                    if arr2.ndim != target_ndim or arr2.shape != target_shape:
                        # skip mismatched frames
                        continue
                    if arr2.ndim == 1:
                        frames.append(go.Frame(data=[go.Scatter(y=arr2, mode='lines')], name=f"frame{i}"))
                    elif arr2.ndim == 2:
                        frames.append(go.Frame(data=[go.Heatmap(z=arr2)], name=f"frame{i}"))

                if not frames:
                    continue

                # Create base figure using the first (consistent) sample
                if target_ndim == 1:
                    fig = go.Figure(data=[go.Scatter(y=_to_np(series[0]), mode='lines')], frames=frames)
                else:
                    fig = go.Figure(data=[go.Heatmap(z=_to_np(series[0]))], frames=frames)

                # Build slider steps referencing the explicit frame names
                steps = []
                for idx in range(len(frames)):
                    steps.append(dict(
                        method='animate',
                        args=[[f"frame{idx}"], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate'}],
                        label=str(idx)
                    ))

                fig.update_layout(
                    title=f"{name} (epoch slider)",
                    updatemenus=[{
                        'type': 'buttons',
                        'buttons': [
                            {'label': 'Play', 'method': 'animate', 'args': [None, {'frame': {'duration': 120, 'redraw': True}, 'fromcurrent': True}]},
                            {'label': 'Pause', 'method': 'animate', 'args': [[None], {'frame': {'duration': 0, 'redraw': False}}]}
                        ]
                    }],
                    sliders=[{
                        'steps': steps,
                        'currentvalue': {'prefix': 'Epoch: '}
                    }]
                )

                fig.write_html(str(base / f"{name}_slider.html"))
            except Exception as e:
                tb = traceback.format_exc()
                print(f"Interactive slider for {name} failed: {type(e).__name__}: {e}\n{tb}")

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

    