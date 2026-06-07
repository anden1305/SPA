#!/usr/bin/env python3
"""HQ-cohort lab comparison plots for cv_quality_cohort_v1 (labs 2, 3, 5).

Reads existing exploration/noise CSVs and optionally loads raw signals for
amplitude CV on HQ mice only. Writes PNGs + summary CSVs under
results/data_exploration/hq_review/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "data_exploration"))

from helpers import (  # noqa: E402
    SLEEP_STAGE_COLORS,
    load_metadata,
    load_participant_data,
)

MANIFEST_PATH = REPO_ROOT / "data/manifests/cv_quality_cohort_v1.yaml"
EXPLORATION_ROOT = REPO_ROOT / "results/data_exploration"
NOISE_ROOT = REPO_ROOT / "results/noise"
OUT_DIR = EXPLORATION_ROOT / "hq_review"

HQ_LABS = ("lab_2", "lab_3", "lab_5")


def load_hq_cohort() -> dict[str, list[str]]:
    with MANIFEST_PATH.open(encoding="utf-8") as f:
        manifest = yaml.safe_load(f)
    return {lab: list(manifest["cohort"][lab]) for lab in HQ_LABS}


def _save(fig: plt.Figure, name: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")
    return path


def plot_50hz_boxplot(hq: dict[str, list[str]]) -> None:
    df = pd.read_csv(EXPLORATION_ROOT / "interference_50hz/interference_50hz_with_outliers.csv")
    hq_ids = {lab: set(ids) for lab, ids in hq.items()}
    mask = df.apply(
        lambda r: r["participant_id"] in hq_ids.get(r["lab"], set()), axis=1
    )
    sub = df.loc[mask].copy()
    sub["log10_ratio"] = np.log10(sub["interference_ratio"].clip(lower=1e-3))

    fig, ax = plt.subplots(figsize=(8, 5))
    order = list(HQ_LABS)
    sns.boxplot(
        data=sub, x="lab", y="log10_ratio", order=order, ax=ax, showfliers=False
    )
    sns.stripplot(
        data=sub, x="lab", y="log10_ratio", order=order, ax=ax,
        color="black", size=2, alpha=0.35, jitter=0.2,
    )
    ax.set_ylabel("log10(50 Hz interference ratio)")
    ax.set_title("HQ cohort: 50 Hz line noise by lab")
    ax.axhline(np.log10(1.0), color="gray", ls="--", lw=0.8)
    _save(fig, "hq_50hz_ratio_boxplot.png")

    rec = (
        sub.groupby(["lab", "participant_id", "run"])["is_outlier"]
        .any()
        .reset_index()
    )
    rate = rec.groupby("lab")["is_outlier"].agg(["sum", "count"]).reset_index()
    rate["outlier_rate"] = rate["sum"] / rate["count"]
    rate.to_csv(OUT_DIR / "hq_50hz_outlier_rate_by_lab.csv", index=False)


def plot_50hz_lab2_mice(hq: dict[str, list[str]]) -> None:
    df = pd.read_csv(EXPLORATION_ROOT / "interference_50hz/interference_50hz_with_outliers.csv")
    hq2 = set(hq["lab_2"])
    sub = df[(df["lab"] == "lab_2") & (df["participant_id"].isin(hq2))].copy()
    worst = (
        sub.groupby("participant_id")["interference_ratio"]
        .max()
        .sort_values(ascending=False)
        .reset_index()
    )
    worst.columns = ["participant_id", "max_interference_ratio"]
    worst.to_csv(OUT_DIR / "hq_lab2_50hz_worst_mice.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 4))
    sns.barplot(
        data=worst, x="participant_id", y="max_interference_ratio", ax=ax,
        color=SLEEP_STAGE_COLORS["NREM"],
    )
    ax.set_yscale("log")
    ax.set_title("HQ lab_2: max 50 Hz ratio per mouse")
    ax.tick_params(axis="x", rotation=45)
    _save(fig, "hq_lab2_50hz_worst_mice.png")


def plot_eeg1_lowfreq(hq: dict[str, list[str]]) -> None:
    frames = []
    for lab_num in (2, 3, 5):
        path = NOISE_ROOT / f"lab_{lab_num}/relative_power_lab{lab_num}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df = df[df["channel"] == "EEG1"].copy()
        lab = f"lab_{lab_num}"
        df = df[df["subject"].isin(hq[lab])]
        df["lab"] = lab
        frames.append(df)
    if not frames:
        return
    sub = pd.concat(frames, ignore_index=True)
    col = "relative_power_0p5_2_over_0p5_30"

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.boxplot(data=sub, x="lab", y=col, order=list(HQ_LABS), ax=ax)
    sns.stripplot(
        data=sub, x="lab", y=col, order=list(HQ_LABS), ax=ax,
        color="black", size=3, alpha=0.4,
    )
    ax.set_ylabel("EEG1: power(0.5–2 Hz) / power(0.5–30 Hz)")
    ax.set_title("HQ cohort: low-frequency EEG1 dominance")
    _save(fig, "hq_eeg1_lowfreq_fraction.png")


def plot_emg_awake_nrem(hq: dict[str, list[str]]) -> None:
    frames = []
    for lab_num in (2, 3, 5):
        path = NOISE_ROOT / f"lab_{lab_num}/emg_rms_awake_over_nrem_lab{lab_num}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        lab = f"lab_{lab_num}"
        df = df[df["subject"].isin(hq[lab])].copy()
        df["lab"] = lab
        frames.append(df)
    if not frames:
        return
    sub = pd.concat(frames, ignore_index=True)
    col = "emg_rms_awake_over_nrem"

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.boxplot(data=sub, x="lab", y=col, order=list(HQ_LABS), ax=ax)
    sns.stripplot(
        data=sub, x="lab", y=col, order=list(HQ_LABS), ax=ax,
        color="black", size=3, alpha=0.4,
    )
    ax.set_ylabel("EMG RMS(Awake) / RMS(NREM)")
    ax.set_title("HQ cohort: EMG muscle-tone contrast")
    _save(fig, "hq_emg_awake_over_nrem.png")


def plot_amplitude_cv_hq(hq: dict[str, list[str]], metadata: pd.DataFrame) -> None:
    """Per-channel amplitude CV on HQ mice (raw processed signals)."""
    rows = []
    for lab in HQ_LABS:
        for pid in hq[lab]:
            runs = metadata.loc[metadata["participant_id"] == pid, "run"].unique()
            for run in runs:
                try:
                    data = load_participant_data(pid, int(run))
                except Exception as exc:  # noqa: BLE001
                    print(f"  skip {pid} run {run}: {exc}")
                    continue
                for ch, arr in data.items():
                    if ch == "labels":
                        continue
                    std = float(np.std(arr))
                    mean = float(np.mean(np.abs(arr)))
                    cv = std / (mean + 1e-12)
                    rows.append({
                        "lab": lab,
                        "participant_id": pid,
                        "run": run,
                        "channel": ch,
                        "cv_amplitude": cv,
                    })
    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "hq_amplitude_cv_detailed.csv", index=False)

    summary = (
        df.groupby(["lab", "channel"])["cv_amplitude"]
        .agg(["median", "mean", "std", "count"])
        .reset_index()
    )
    summary.to_csv(OUT_DIR / "hq_amplitude_cv_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    channels = sorted(df["channel"].unique(), key=lambda c: (c != "EMG", c))
    plot_df = df[df["channel"].isin(channels)]
    sns.boxplot(
        data=plot_df, x="channel", y="cv_amplitude", hue="lab",
        order=channels, hue_order=list(HQ_LABS), ax=ax,
    )
    ax.set_yscale("log")
    ax.set_title("HQ cohort: amplitude CV by channel (log scale)")
    ax.legend(title="Lab", loc="upper right")
    _save(fig, "hq_amplitude_cv_by_channel.png")


def plot_transition_churn(hq: dict[str, list[str]]) -> None:
    path = EXPLORATION_ROOT / "stage_transitions/per_record_transition_summary.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    hq_ids = {lab: set(ids) for lab, ids in hq.items()}
    mask = df.apply(
        lambda r: r["participant_id"] in hq_ids.get(r["lab"], set()), axis=1
    )
    sub = df.loc[mask & df["lab"].isin(HQ_LABS)].copy()
    sub["change_rate"] = sub["total_changes"] / sub["total_transitions"].clip(lower=1)

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.boxplot(data=sub, x="lab", y="change_rate", order=list(HQ_LABS), ax=ax)
    sns.stripplot(
        data=sub, x="lab", y="change_rate", order=list(HQ_LABS), ax=ax,
        color="black", size=3, alpha=0.4,
    )
    ax.set_ylabel("Stage changes / total 4 s epochs")
    ax.set_title("HQ cohort: label transition churn (proxy for scoring blur)")
    _save(fig, "hq_stage_transition_churn.png")

    sub.groupby("lab")["change_rate"].agg(["median", "mean", "count"]).to_csv(
        OUT_DIR / "hq_stage_transition_churn_summary.csv"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-amplitude-cv",
        action="store_true",
        help="Skip raw-signal CV plot (loads .npy per HQ recording).",
    )
    args = parser.parse_args()

    print("HQ lab review — loading cohort manifest")
    hq = load_hq_cohort()
    for lab, mice in hq.items():
        print(f"  {lab}: {len(mice)} mice")

    plot_50hz_boxplot(hq)
    plot_50hz_lab2_mice(hq)
    plot_eeg1_lowfreq(hq)
    plot_emg_awake_nrem(hq)
    plot_transition_churn(hq)

    if not args.skip_amplitude_cv:
        print("Computing amplitude CV on HQ recordings...")
        metadata = load_metadata()
        plot_amplitude_cv_hq(hq, metadata)

    print(f"Done — outputs in {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
