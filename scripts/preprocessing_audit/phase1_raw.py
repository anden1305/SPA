"""Phase 1: raw data audit before VAE preprocessing."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import signal

from scripts.data_exploration.helpers import (
    SLEEP_STAGE_COLORS,
    SLEEP_STAGE_MAPPING,
    STAGE_ORDER,
    load_metadata,
    load_participant_data,
)
from scripts.preprocessing_audit.manifest import LABS, RunRecord, iter_runs, load_manifest

FS = 128
EPOCH_SAMPLES = FS * 4
BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


def _cohort_mouse_ids(manifest: dict) -> set[str]:
    return set(manifest["inventory"].keys())


def _epoch_labels(labels: np.ndarray) -> np.ndarray:
    n_epochs = len(labels) // EPOCH_SAMPLES
    if n_epochs == 0:
        return labels
    trimmed = labels[: n_epochs * EPOCH_SAMPLES].reshape(n_epochs, EPOCH_SAMPLES)
    out = []
    for row in trimmed:
        vals, counts = np.unique(row, return_counts=True)
        out.append(int(vals[counts.argmax()]))
    return np.array(out, dtype=np.int64)


def _band_power_epoch(signal_1d: np.ndarray, band: tuple[float, float]) -> float:
    f, pxx = signal.welch(signal_1d, fs=FS, nperseg=min(256, len(signal_1d)))
    mask = (f >= band[0]) & (f <= band[1])
    if not mask.any():
        return 0.0
    return float(np.trapz(pxx[mask], f[mask]))


def _rel_50hz(signal_1d: np.ndarray) -> float:
    f, pxx = signal.welch(signal_1d, fs=FS, nperseg=min(256, len(signal_1d)))
    total = np.trapz(pxx, f) + 1e-12
    mask = (f >= 49) & (f <= 51)
    line = np.trapz(pxx[mask], f[mask]) if mask.any() else 0.0
    return float(line / total)


def collect_per_mouse_rows(manifest: dict) -> pd.DataFrame:
    cohort = _cohort_mouse_ids(manifest)
    rows = []
    for rec in iter_runs(manifest):
        try:
            data = load_participant_data(rec.participant_id, rec.run)
        except Exception:
            continue
        labels = data["labels"]
        epoch_lbl = _epoch_labels(labels)
        n_epochs = len(epoch_lbl)
        stage_pct = {}
        for stage_name in STAGE_ORDER:
            stage_pct[stage_name] = 0.0
        for lid in np.unique(epoch_lbl):
            name = SLEEP_STAGE_MAPPING.get(int(lid), "Unknown")
            if name in stage_pct:
                stage_pct[name] = float((epoch_lbl == lid).mean() * 100)

        rms_vals = []
        for sig in rec.signals:
            if sig in data:
                rms_vals.append(float(np.sqrt(np.mean(data[sig] ** 2))))
        mean_rms = float(np.mean(rms_vals)) if rms_vals else np.nan

        hz50 = []
        for sig in rec.signals:
            if sig in data:
                hz50.append(_rel_50hz(np.asarray(data[sig], dtype=float)))
        mean_50 = float(np.mean(hz50)) if hz50 else np.nan

        wake_var_proxy = np.nan
        if "EEG1" in data:
            wake_mask = epoch_lbl == 0
            if wake_mask.sum() > 5:
                eeg = data["EEG1"]
                chunks = []
                for i, is_wake in enumerate(wake_mask):
                    if is_wake:
                        seg = eeg[i * EPOCH_SAMPLES : (i + 1) * EPOCH_SAMPLES]
                        chunks.append(np.var(seg))
                if chunks:
                    wake_var_proxy = float(np.median(chunks))

        rows.append(
            {
                "participant_id": rec.participant_id,
                "lab": rec.lab,
                "run": rec.run,
                "n_epochs": n_epochs,
                "hours": rec.hours,
                "pct_awake": stage_pct.get("Awake", 0),
                "pct_nrem": stage_pct.get("NREM", 0),
                "pct_rem": stage_pct.get("REM", 0),
                "pct_artifact": stage_pct.get("Artifact", 0),
                "mean_rms": mean_rms,
                "rel_50hz": mean_50,
                "wake_variance_proxy": wake_var_proxy,
            }
        )
    return pd.DataFrame(rows)


def _plot_lab_aggregates(df: pd.DataFrame, raw_dir: Path) -> None:
    for lab in LABS:
        sub = df[df["lab"] == lab]
        if sub.empty:
            continue
        lab_dir = raw_dir / lab
        lab_dir.mkdir(parents=True, exist_ok=True)

        # label composition
        fig, ax = plt.subplots(figsize=(10, 5))
        mice = sub["participant_id"] + "_r" + sub["run"].astype(str)
        bottom = np.zeros(len(sub))
        for stage, col, color in [
            ("Awake", "pct_awake", SLEEP_STAGE_COLORS["Awake"]),
            ("NREM", "pct_nrem", SLEEP_STAGE_COLORS["NREM"]),
            ("REM", "pct_rem", SLEEP_STAGE_COLORS["REM"]),
            ("Artifact", "pct_artifact", SLEEP_STAGE_COLORS["Artifact"]),
        ]:
            vals = sub[col].values
            ax.bar(mice, vals, bottom=bottom, label=stage, color=color)
            bottom += vals
        ax.set_title(f"Stage composition — {lab}")
        ax.set_ylabel("% epochs")
        ax.legend(loc="upper right", fontsize=8)
        plt.xticks(rotation=90, fontsize=6)
        plt.tight_layout()
        plt.savefig(lab_dir / "label_composition.png", dpi=150)
        plt.close()

        # amplitude
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.boxplot(data=sub, x="participant_id", y="mean_rms", ax=ax)
        ax.set_title(f"Mean channel RMS — {lab}")
        plt.xticks(rotation=45, ha="right", fontsize=7)
        plt.tight_layout()
        plt.savefig(lab_dir / "amplitude_distributions.png", dpi=150)
        plt.close()

    # band power by lab (theta/delta on EEG1)
    band_rows = []
    for rec in iter_runs(load_manifest()):
        if rec.lab not in LABS:
            continue
        try:
            data = load_participant_data(rec.participant_id, rec.run)
        except Exception:
            continue
        if "EEG1" not in data:
            continue
        eeg = np.asarray(data["EEG1"], dtype=float)
        epoch_lbl = _epoch_labels(data["labels"])
        for lid in np.unique(epoch_lbl):
            mask_epochs = epoch_lbl == lid
            if mask_epochs.sum() < 3:
                continue
            segs = []
            for i, m in enumerate(mask_epochs):
                if m:
                    segs.append(eeg[i * EPOCH_SAMPLES : (i + 1) * EPOCH_SAMPLES])
            if not segs:
                continue
            sig = np.concatenate(segs)
            row = {
                "lab": rec.lab,
                "participant_id": rec.participant_id,
                "stage": SLEEP_STAGE_MAPPING.get(int(lid), "?"),
            }
            for bname, brange in BANDS.items():
                row[bname] = _band_power_epoch(sig, brange)
            band_rows.append(row)
    if band_rows:
        bdf = pd.DataFrame(band_rows)
        fig, ax = plt.subplots(figsize=(10, 5))
        plot_df = bdf.groupby(["lab", "stage"])["theta"].mean().reset_index()
        sns.barplot(data=plot_df, x="stage", y="theta", hue="lab", ax=ax, order=[s for s in STAGE_ORDER if s != "Artifact"])
        ax.set_title("Mean theta power (EEG1) by lab and stage")
        plt.tight_layout()
        for lab in LABS:
            (raw_dir / lab).mkdir(parents=True, exist_ok=True)
            plt.savefig(raw_dir / lab / "band_power_by_lab.png", dpi=150)
        plt.close()

        # PSD by stage per lab
        for lab in LABS:
            lab_recs = [r for r in iter_runs(load_manifest()) if r.lab == lab][:3]
            fig, ax = plt.subplots(figsize=(8, 5))
            for rec in lab_recs:
                try:
                    data = load_participant_data(rec.participant_id, rec.run)
                except Exception:
                    continue
                if "EEG1" not in data:
                    continue
                eeg = np.asarray(data["EEG1"], dtype=float)[: FS * 3600]
                f, pxx = signal.welch(eeg, fs=FS, nperseg=512)
                ax.plot(f, 10 * np.log10(pxx + 1e-12), alpha=0.5, label=rec.participant_id)
            ax.set_xlim(0, 45)
            ax.set_title(f"Example PSD (EEG1) — {lab}")
            ax.set_xlabel("Hz")
            ax.legend(fontsize=7)
            plt.tight_layout()
            plt.savefig(raw_dir / lab / "psd_by_stage.png", dpi=150)
            plt.close()


def write_summary_raw(df: pd.DataFrame, raw_dir: Path) -> None:
    lines = ["# Raw data audit summary\n"]
    risks = []
    for lab in LABS:
        sub = df[df["lab"] == lab]
        lines.append(f"## {lab}\n")
        if sub.empty:
            lines.append("- No data.\n")
            continue
        lines.append(f"- Mice/runs: {len(sub)} recordings.\n")
        lines.append(f"- Median %NREM: {sub['pct_nrem'].median():.1f}, %REM: {sub['pct_rem'].median():.1f}.\n")
        if lab in ("lab_3", "lab_5"):
            lines.append(f"- Median %Artifact: {sub['pct_artifact'].median():.1f}.\n")
        else:
            lines.append(f"- No Artifact label; median wake variance proxy: {sub['wake_variance_proxy'].median():.3e}.\n")
        lines.append(f"- Median RMS: {sub['mean_rms'].median():.3f}, 50Hz rel power: {sub['rel_50hz'].median():.4f}.\n")
        if lab == "lab_2" and sub["wake_variance_proxy"].median() > df[df["lab"] == "lab_3"]["wake_variance_proxy"].median():
            risks.append(("Artifact handling asymmetry (lab_2)", 1))
        if sub["mean_rms"].std() / (sub["mean_rms"].mean() + 1e-8) > 0.4:
            risks.append((f"Amplitude spread within {lab}", 2))
    lines.append("\n## Ranked risks for VAE if unchanged\n\n")
    for name, rank in sorted(set(risks), key=lambda x: x[1]):
        lines.append(f"{rank}. {name}\n")
    if not risks:
        lines.append("1. Cross-lab montage (EEG1/EEG3 vs EEG1/EEG2)\n2. post_normalize per-run not per-lab\n")
    (raw_dir / "SUMMARY_raw.md").write_text("".join(lines), encoding="utf-8")


def run_phase1(manifest: dict, out_dir: Path) -> pd.DataFrame:
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    df = collect_per_mouse_rows(manifest)
    tables = out_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    df.to_csv(tables / "per_mouse_summary.csv", index=False)
    _plot_lab_aggregates(df, raw_dir)
    write_summary_raw(df, raw_dir)
    return df
