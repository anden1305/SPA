"""Phase 2: model-input plots and normalization scope audit."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA

from scripts.preprocessing_audit.config_utils import load_global_config
from scripts.preprocessing_audit.data_loading import (
    load_mssv_array,
    run_vae_preprocessing,
    subsample_sequences,
)
from scripts.preprocessing_audit.manifest import LABS, RunRecord, iter_runs, mice_by_lab
from scripts.preprocessing_audit.metrics import (
    compute_metrics,
    counterfactual_post_normalize,
    post_norm_stats_run_level,
)

STAGE_NAMES = {0: "Awake", 1: "NREM", 2: "REM", 3: "Artifact"}


def _preprocess_no_post(cfg, x, y):
    from scripts.preprocessing_audit.config_utils import apply_overrides

    cfg2 = apply_overrides(cfg, {"cvae.post_normalize": False})
    return run_vae_preprocessing(cfg2, x, y)


def process_all_runs(
    cfg,
    manifest: dict,
    max_sequences: int = 400,
) -> tuple[pd.DataFrame, dict[str, list], dict]:
    rows = []
    by_lab: dict[str, list] = {lab: [] for lab in LABS}
    norm_rows = []
    cache_x: dict[tuple[str, int], np.ndarray] = {}

    for rec in iter_runs(manifest):
        try:
            x_raw, y_raw, meta = load_mssv_array(rec)
            x_feat, y_feat = run_vae_preprocessing(cfg, x_raw, y_raw)
            x_nopost, _ = _preprocess_no_post(cfg, x_raw, y_raw)
        except Exception as exc:
            print(f"skip {rec.participant_id} run {rec.run}: {exc}")
            continue

        if x_feat.shape[0] == 0:
            continue
        x_sub, y_sub = subsample_sequences(x_feat, y_feat, max_sequences)
        cache_x[(rec.participant_id, rec.run)] = (x_sub, y_sub, meta)

        mets = compute_metrics(x_sub, y_sub)
        mets.update(
            {
                "participant_id": rec.participant_id,
                "lab": rec.lab,
                "run": rec.run,
                "n_timesteps": meta["n_timesteps_after"],
            }
        )
        rows.append(mets)
        by_lab[rec.lab].append(x_sub.reshape(len(x_sub), -1))

        mean_r, std_r = post_norm_stats_run_level(x_nopost)
        norm_rows.append(
            {
                "participant_id": rec.participant_id,
                "lab": rec.lab,
                "run": rec.run,
                "postnorm_mean_abs": float(np.mean(np.abs(mean_r))),
                "postnorm_std_mean": float(np.mean(std_r)),
            }
        )

    return pd.DataFrame(rows), by_lab, {"cache": cache_x, "norm": pd.DataFrame(norm_rows)}


def _plot_model_input_example(x: np.ndarray, y: np.ndarray, path: Path, title: str) -> None:
    """x: (n_seq, S, C, F) — plot first sequence."""
    if x.shape[0] == 0:
        return
    seq_x = x[0]
    seq_y = y[0]
    S, C, F = seq_x.shape
    fig, axes = plt.subplots(C + 1, 1, figsize=(12, 2 * (C + 1)), sharex=True)
    if C == 1:
        axes = [axes]
    for c in range(C):
        axes[c].imshow(seq_x[:, c, :].T, aspect="auto", origin="lower", cmap="viridis")
        axes[c].set_ylabel(f"ch{c}")
        axes[c].set_title(f"{title} — channel {c}")
    stage_colors = {0: "#79C780", 1: "#6C9BD9", 2: "#D1779A", 3: "#FFC675"}
    colors = [stage_colors.get(int(v), "#888") for v in seq_y]
    axes[-1].scatter(range(S), seq_y, c=colors, s=8)
    axes[-1].set_ylabel("stage")
    axes[-1].set_xlabel(f"sequence positions (S={S}, C={C}, F={F})")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _plot_pca_by_lab(by_lab: dict[str, list], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for lab in LABS:
        arrs = by_lab.get(lab, [])
        if not arrs:
            continue
        X = np.vstack(arrs)
        if len(X) > 3000:
            idx = np.random.default_rng(0).choice(len(X), 3000, replace=False)
            X = X[idx]
        pca = PCA(2)
        Z = pca.fit_transform(X)
        ax.scatter(Z[:, 0], Z[:, 1], alpha=0.15, s=4, label=lab)
    ax.legend()
    ax.set_title("PCA of flattened model inputs (unlabeled mix)")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _plot_fft_by_stage(cache: dict, lab: str, path: Path) -> None:
    """Mean spectrum per stage from first available mouse."""
    for (pid, run), (x, y, meta) in cache.items():
        if meta["lab"] != lab:
            continue
        # x: n_seq, S, C, F
        seq_x = x[0]
        seq_y = y[0]
        C, F = seq_x.shape[1], seq_x.shape[2]
        fig, axes = plt.subplots(1, C, figsize=(4 * C, 3))
        if C == 1:
            axes = [axes]
        for c in range(C):
            for stage_id, name in STAGE_NAMES.items():
                mask = seq_y == stage_id
                if not mask.any():
                    continue
                spec = seq_x[mask, c, :].mean(axis=0)
                axes[c].plot(spec, label=name)
            axes[c].set_title(f"ch{c}")
            axes[c].legend(fontsize=7)
        fig.suptitle(f"log-power features by stage — {lab} ({pid})")
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        return


def _plot_lab_contrast(cache: dict, out_path: Path) -> None:
    """Side-by-side model input for lab_3 vs lab_2/5 (first seq heatmap ch0)."""
    ref_lab = "lab_3"
    ref = None
    others = []
    for key, (x, y, meta) in cache.items():
        if meta["lab"] == ref_lab and ref is None:
            ref = (key, x, y)
        elif meta["lab"] in ("lab_2", "lab_5"):
            others.append((key, x, y, meta["lab"]))
    if ref is None or not others:
        return
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    key, x, y = ref
    axes[0, 0].imshow(x[0, :, 0, :].T, aspect="auto", origin="lower", cmap="viridis")
    axes[0, 0].set_title(f"{ref_lab} {key[0]} ch0")
    for i, (key2, x2, y2, lab) in enumerate(others[:3]):
        r, c = (i + 1) // 2, (i + 1) % 2
        axes[r, c].imshow(x2[0, :, 0, :].T, aspect="auto", origin="lower", cmap="viridis")
        axes[r, c].set_title(f"{lab} {key2[0]} ch0")
    plt.suptitle("Model input contrast (first sequence, channel 0)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _normalization_scope_plots(norm_df: pd.DataFrame, cache: dict, cfg, out_dir: Path) -> None:
    ns_dir = out_dir / "normalization_scope"
    ns_dir.mkdir(parents=True, exist_ok=True)

    if not norm_df.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        for run in sorted(norm_df["run"].unique()):
            sub = norm_df[norm_df["run"] == run]
            ax.scatter(sub["participant_id"], sub["postnorm_std_mean"], label=f"run {run}", alpha=0.7)
        ax.set_title("Pre-post_norm std per run (mean across features; used for z-score)")
        ax.tick_params(axis="x", rotation=90)
        ax.legend()
        plt.tight_layout()
        plt.savefig(ns_dir / "postnorm_scale_by_run.png", dpi=150)
        plt.close()

    # counterfactual: run-level vs lab-pooled stats on no-post features
    lab_features: dict[str, list[np.ndarray]] = {lab: [] for lab in LABS}
    run_features: list[np.ndarray] = []
    for (pid, run), (x, y, meta) in cache.items():
        try:
            x_raw, y_raw, _ = load_mssv_array(
                RunRecord(pid, meta["lab"], run, meta["signals"], 0.0)
            )
            x_nopost, _ = _preprocess_no_post(cfg, x_raw, y_raw)
            if x_nopost.size == 0:
                continue
            flat = x_nopost.reshape(x_nopost.shape[0], -1)
            if len(flat) > 200:
                flat = flat[np.random.default_rng(0).choice(len(flat), 200, replace=False)]
            lab_features[meta["lab"]].append(flat)
            run_features.append(flat)
        except Exception:
            continue

    deltas = []
    for lab in LABS:
        if not lab_features[lab]:
            continue
        X_lab = np.vstack(lab_features[lab])
        mean_lab = np.mean(X_lab, axis=0, keepdims=True)
        std_lab = np.std(X_lab, axis=0, keepdims=True) + 1e-8
        for (pid, run), (x, y, meta) in cache.items():
            if meta["lab"] != lab:
                continue
            try:
                x_raw, y_raw, _ = load_mssv_array(
                    RunRecord(pid, meta["lab"], run, meta["signals"], 0.0)
                )
                x_nopost, _ = _preprocess_no_post(cfg, x_raw, y_raw)
                mean_run, std_run = post_norm_stats_run_level(x_nopost)
                z_run = counterfactual_post_normalize(x_nopost, mean_run, std_run)
                z_lab = counterfactual_post_normalize(x_nopost, mean_lab, std_lab)
                delta = float(np.mean(np.abs(z_run - z_lab)))
                deltas.append({"lab": lab, "participant_id": pid, "run": run, "mean_abs_delta": delta})
            except Exception:
                continue

    tables_dir = out_dir.parent / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    scope_rows = norm_df.to_dict("records") if not norm_df.empty else []
    for row in deltas:
        row["metric"] = "mean_abs_delta_run_vs_lab_postnorm"
        scope_rows.append(row)
    if not scope_rows and not norm_df.empty:
        scope_rows = norm_df.assign(metric="postnorm_std_per_run").to_dict("records")
    pd.DataFrame(scope_rows).to_csv(tables_dir / "normalization_scope.csv", index=False)

    if deltas:
        ddf = pd.DataFrame(deltas)
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.boxplot(data=ddf, x="lab", y="mean_abs_delta", ax=ax)
        ax.set_title("Counterfactual: run-level vs lab-pooled post-norm |delta|")
        plt.tight_layout()
        plt.savefig(ns_dir / "counterfactual_lab_vs_run_norm.png", dpi=150)
        plt.close()
    else:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No counterfactual deltas computed", ha="center")
        ax.axis("off")
        plt.savefig(ns_dir / "counterfactual_lab_vs_run_norm.png", dpi=150)
        plt.close()


def _plot_window_counts(metrics_df: pd.DataFrame, lab: str, path: Path) -> None:
    sub = metrics_df[metrics_df["lab"] == lab]
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(sub["participant_id"] + "_r" + sub["run"].astype(str), sub["n_sequences"])
    ax.set_title(f"Sequence counts after preprocessing — {lab}")
    plt.xticks(rotation=90, fontsize=6)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _feature_pca_labeled(cache: dict, lab: str, path: Path) -> None:
    feats, labels = [], []
    for (pid, run), (x, y, meta) in cache.items():
        if meta["lab"] != lab:
            continue
        n = min(50, x.shape[0])
        for i in range(n):
            row = y[i]
            vals, counts = np.unique(row, return_counts=True)
            feats.append(x[i].reshape(-1))
            labels.append(int(vals[counts.argmax()]))
    if len(feats) < 10:
        return
    X = np.stack(feats)
    Z = PCA(2).fit_transform(X)
    fig, ax = plt.subplots(figsize=(7, 5))
    for lid in np.unique(labels):
        m = labels == lid
        ax.scatter(Z[m, 0], Z[m, 1], s=10, alpha=0.5, label=STAGE_NAMES.get(lid, str(lid)))
    ax.legend()
    ax.set_title(f"Feature PCA by stage — {lab}")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def write_summary_after(metrics_df: pd.DataFrame, after_dir: Path) -> None:
    lines = ["# Model-input audit summary\n\n"]
    for lab in LABS:
        sub = metrics_df[metrics_df["lab"] == lab]
        lines.append(f"## {lab}\n")
        if sub.empty:
            lines.append("- No data.\n")
            continue
        sil = sub["silhouette_k3"].median()
        lines.append(f"- Median silhouette (stage separability): {sil:.3f}\n")
        lines.append(f"- Median sequences per run: {sub['n_sequences'].median():.0f}\n")
        generalize = "yes" if sil >= 0.15 and not np.isnan(sil) else "uncertain/no"
        if lab != "lab_3" and not np.isnan(sil):
            ref = metrics_df[metrics_df["lab"] == "lab_3"]["silhouette_k3"].median()
            if not np.isnan(ref) and sil < ref - 0.05:
                generalize = "no"
        lines.append(
            f"- Expect subject-only VAE to generalize from lab_3-looking inputs? **{generalize}**\n"
        )
    (after_dir / "SUMMARY_after.md").write_text("".join(lines), encoding="utf-8")


def run_phase2(cfg, manifest: dict, out_dir: Path, max_sequences: int = 400) -> pd.DataFrame:
    after_dir = out_dir / "after"
    after_dir.mkdir(parents=True, exist_ok=True)
    metrics_df, by_lab, aux = process_all_runs(cfg, manifest, max_sequences)
    cache = aux["cache"]
    norm_df = aux["norm"]

    tables = out_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    if not metrics_df.empty:
        metrics_df.to_csv(tables / "feature_stats_by_lab.csv", index=False)

    # pick example from lab_3
    example = None
    for key, pack in cache.items():
        if pack[2]["lab"] == "lab_3":
            example = pack
            break
    if example:
        x, y, _ = example
        (after_dir / "lab_3").mkdir(parents=True, exist_ok=True)
        _plot_model_input_example(x, y, after_dir / "lab_3" / "model_input_examples.png", "lab_3 example")

    for lab in LABS:
        lab_dir = after_dir / lab
        lab_dir.mkdir(parents=True, exist_ok=True)
        _plot_fft_by_stage(cache, lab, lab_dir / "fft_logpower_by_stage.png")
        _feature_pca_labeled(cache, lab, lab_dir / "feature_pca_triple.png")
        _plot_window_counts(metrics_df, lab, lab_dir / "window_count_diagnostics.png")
        # per-channel spectrum: mean over sequences ch0
        ch_specs = []
        for (pid, run), (x, y, meta) in cache.items():
            if meta["lab"] != lab:
                continue
            ch_specs.append(x[:, :, 0, :].mean(axis=(0, 1)))
        if ch_specs:
            fig, ax = plt.subplots(figsize=(8, 4))
            for i, spec in enumerate(ch_specs[:8]):
                ax.plot(spec, alpha=0.5, label=f"mouse{i}")
            ax.set_title(f"Mean log-power ch0 — {lab}")
            plt.tight_layout()
            plt.savefig(lab_dir / "per_channel_spectrum.png", dpi=150)
            plt.close()

    (after_dir / "lab_3").mkdir(parents=True, exist_ok=True)
    _plot_lab_contrast(cache, after_dir / "lab_3" / "model_input_lab_contrast.png")
    cross_lab_dir = after_dir / "cross_lab"
    cross_lab_dir.mkdir(parents=True, exist_ok=True)
    _plot_pca_by_lab(by_lab, cross_lab_dir / "pca_all_labs_unlabeled.png")
    _normalization_scope_plots(norm_df, cache, cfg, after_dir)
    write_summary_after(metrics_df, after_dir)
    return metrics_df
