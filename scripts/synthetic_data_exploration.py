"""Quick exploration of generated synthetic sleep data.

Loads a saved synthetic run from data/synthetic (default prefix 'synthetic_default')
and produces basic summaries:
- Label counts & transition matrix
- Per-epoch standard deviation distribution
- Mean power spectrum per class
Outputs figures into results/synthetic_exploration.
"""
from __future__ import annotations
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

SAVE_DIR = Path("data") / "synthetic"
RESULTS = Path("results") / "synthetic_exploration"
RESULTS.mkdir(parents=True, exist_ok=True)

def load_run(prefix: str = "synthetic_default"):
    eeg_path = SAVE_DIR / f"{prefix}_eeg.npy"
    labels_path = SAVE_DIR / f"{prefix}_labels.npy"
    meta_path = SAVE_DIR / f"{prefix}_metadata.json"
    if not eeg_path.exists():
        raise FileNotFoundError(eeg_path)
    eeg = np.load(eeg_path)
    labels = np.load(labels_path)
    import json
    meta = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
    return eeg, labels, meta

def summarize_labels(labels: np.ndarray, meta: dict):
    stages = meta.get("stages", [])
    unique, counts = np.unique(labels, return_counts=True)
    mapping = {u: stages[u] if u < len(stages) else str(u) for u in unique}
    summary_lines = []
    for u,c in zip(unique, counts):
        summary_lines.append(f"{u} ({mapping[u]}): {c} ({c/len(labels):.2%})")
    (RESULTS / "label_summary.txt").write_text("\n".join(summary_lines))
    # Bar plot
    plt.figure(figsize=(5,3))
    # Use hue to avoid future deprecation warning
    stage_names = [mapping[u] for u in unique]
    import pandas as pd
    df_counts = pd.DataFrame({"stage": stage_names, "count": counts})
    sns.barplot(data=df_counts, x="stage", y="count", hue="stage", palette="viridis", legend=False)
    plt.title("Stage counts")
    plt.ylabel("Epochs")
    plt.tight_layout()
    plt.savefig(RESULTS / "stage_counts.png", dpi=150)
    plt.close()
    # Transition matrix
    tm = np.zeros((len(stages), len(stages)))
    for a,b in zip(labels[:-1], labels[1:]):
        tm[a,b]+=1
    row_sums = tm.sum(axis=1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        tm_prob = np.where(row_sums>0, tm/row_sums, 0)
    plt.figure(figsize=(4,4))
    sns.heatmap(tm_prob, annot=True, fmt=".2f", cmap="magma", xticklabels=stages, yticklabels=stages)
    plt.title("Transition matrix (P(next|current))")
    plt.tight_layout()
    plt.savefig(RESULTS / "transition_matrix.png", dpi=150)
    plt.close()

def epoch_std_distribution(eeg: np.ndarray):
    stds = eeg.std(axis=1)
    unique_vals = np.unique(stds)
    plt.figure(figsize=(5,3))
    if len(unique_vals) <= 1:
        # Degenerate case: all epochs identical amplitude
        val = float(unique_vals[0]) if unique_vals.size else 0.0
        plt.axvline(val, color="steelblue")
        plt.text(val, 0.5, f"std={val:.3g}", rotation=90, va="center")
        plt.xlim(val - 1 if val != 0 else -1, val + 1 if val != 0 else 1)
        plt.ylim(0,1)
    else:
        n_bins = min(30, len(unique_vals))
        # Guard against zero range
        data_range = stds.max() - stds.min()
        if data_range <= 1e-12:
            n_bins = 1
        sns.histplot(stds, bins=n_bins, kde=len(unique_vals) > 5)
    plt.xlabel("Epoch std (a.u.)")
    plt.title("Epoch amplitude distribution")
    plt.tight_layout()
    plt.savefig(RESULTS / "epoch_std_hist.png", dpi=150)
    plt.close()
    np.savetxt(RESULTS / "epoch_std_summary.txt", [stds.mean(), np.median(stds), stds.min(), stds.max()], header="mean,median,min,max")

def mean_power_spectrum(eeg: np.ndarray, labels: np.ndarray, meta: dict):
    sr = meta.get("sampling_rate_hz", meta.get("config_used", {}).get("sampling_rate_hz", 1))
    n = eeg.shape[1]
    freqs = np.fft.rfftfreq(n, d=1/sr)
    stages = meta.get("stages", [])
    plt.figure(figsize=(7,4))
    for idx, stage in enumerate(stages):
        mask = labels == idx
        if not mask.any():
            continue
        spec = np.abs(np.fft.rfft(eeg[mask], axis=1))**2
        mean_spec = spec.mean(axis=0)
        plt.plot(freqs, mean_spec, label=stage)
    plt.xlim(0, 60)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power (a.u.)")
    plt.title("Mean power spectrum per stage")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS / "mean_power_spectrum.png", dpi=150)
    plt.close()

def main():
    eeg, labels, meta = load_run()
    summarize_labels(labels, meta)
    epoch_std_distribution(eeg)
    mean_power_spectrum(eeg, labels, meta)
    print("Synthetic exploration complete ->", RESULTS)

if __name__ == "__main__":
    main()
