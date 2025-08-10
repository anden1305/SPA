"""Data exploration for the ds006366 sleep EEG dataset.

Generates summary statistics and a range of plots:
1. Participant distribution per lab.
2. Sleep stage distribution (overall + per participant + per lab).
3. Hypnogram example (first subject) aggregated to 30 s epochs.
4. Episode length distribution per stage.
5. Transition matrix between stages.
6. Recording duration distribution.

Run directly: uv run python notebooks/data_exploration.py
Outputs figures to results/data_exploration/.
"""

from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass
import itertools
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA_ROOT = Path("data") / "ds006366"
OUTPUT_DIR = Path("results") / "data_exploration"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Mapping numeric stage codes (guessed from typical sleep staging) – adjust if needed
STAGE_MAP = {
    0: "Unknown",
    1: "N1",
    2: "N2",
    3: "N3",
    4: "REM",
    5: "Wake",  # If wake present (not seen yet)
}


def load_participants() -> pd.DataFrame:
    path = DATA_ROOT / "participants.tsv"
    df = pd.read_csv(path, sep="\t")
    return df


def iter_event_files():
    for subj_dir in sorted(DATA_ROOT.glob("sub-*")):
        events = list((subj_dir / "eeg").glob("*_events.tsv"))
        for ev_path in events:
            yield ev_path


def load_events(ev_path: Path) -> pd.DataFrame:
    df = pd.read_csv(ev_path, sep="\t")
    # ensure expected cols
    assert {"onset", "duration", "stage"}.issubset(df.columns)
    return df


def aggregate_events(df: pd.DataFrame, epoch_len: int = 30) -> pd.DataFrame:
    """Aggregate 4 s micro-epochs into 30 s epochs via majority vote.

    Assumes constant duration per row (observed 4 s). Adjust if variable.
    """
    d = df.copy()
    base = d["onset"].min()
    d["epoch_index"] = ((d["onset"] - base) // epoch_len).astype(int)
    # majority stage per epoch
    agg = (
        d.groupby("epoch_index")
        .agg(
            onset=("onset", "min"),
            duration=("duration", "sum"),
            stage=("stage", lambda x: x.value_counts().idxmax()),
        )
        .reset_index(drop=True)
    )
    return agg


def compute_transition_matrix(stages: list[int]) -> pd.DataFrame:
    uniq = sorted(set(stages))
    idx = uniq
    counts = pd.DataFrame(0, index=idx, columns=idx, dtype=int)
    for a, b in zip(stages[:-1], stages[1:]):
        counts.loc[a, b] += 1
    probs = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0)
    counts.index.name = counts.columns.name = "from"
    probs.index.name = probs.columns.name = "from"
    return probs.fillna(0)


def stage_labelize(series: pd.Series) -> pd.Series:
    return series.map(STAGE_MAP).fillna(series.astype(str))


def plot_participant_distribution(participants: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6,4))
    order = participants["lab"].value_counts().index
    sns.countplot(data=participants, x="lab", order=order, ax=ax, palette="viridis")
    ax.set_title("Participants per lab")
    ax.set_ylabel("Count")
    for p in ax.patches:
        ax.annotate(int(p.get_height()), (p.get_x()+p.get_width()/2, p.get_height()), ha='center', va='bottom', fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "participants_per_lab.png", dpi=150)
    plt.close(fig)


def plot_stage_distribution(stage_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6,4))
    order = stage_df["stage_label"].value_counts().index
    sns.countplot(data=stage_df, x="stage_label", order=order, ax=ax, palette="magma")
    ax.set_title("Micro-epoch stage counts (all subjects)")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "stage_distribution_overall.png", dpi=150)
    plt.close(fig)

    # per lab
    fig, ax = plt.subplots(figsize=(8,5))
    lab_counts = stage_df.groupby(["lab", "stage_label"]).size().reset_index(name="count")
    sns.barplot(data=lab_counts, x="lab", y="count", hue="stage_label", ax=ax)
    ax.set_title("Stage distribution per lab")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "stage_distribution_per_lab.png", dpi=150)
    plt.close(fig)


def plot_hypnogram(example_epochs: pd.DataFrame, subject: str):
    fig, ax = plt.subplots(figsize=(10,3))
    ax.step(example_epochs["onset"] / 3600, example_epochs["stage"], where="post")
    ax.set_xlabel("Hours from lights off")
    ax.set_ylabel("Stage (code)")
    ax.set_title(f"Hypnogram (30 s epochs) {subject}")
    ax.set_yticks(sorted(STAGE_MAP.keys()))
    ax.set_yticklabels([STAGE_MAP[k] for k in sorted(STAGE_MAP.keys())])
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"hypnogram_{subject}.png", dpi=150)
    plt.close(fig)


def plot_episode_lengths(stage_df: pd.DataFrame):
    # Define episodes: consecutive identical stage codes within same subject
    records = []
    for (subj), df_sub in stage_df.groupby("subject"):
        df_sub = df_sub.sort_values("onset")
        prev_stage = None
        run_len = 0
        start_onset = None
        for row in df_sub.itertuples():
            if row.stage != prev_stage:
                if prev_stage is not None:
                    records.append({
                        "subject": subj,
                        "stage": prev_stage,
                        "start_onset": start_onset,
                        "duration": run_len,
                    })
                prev_stage = row.stage
                run_len = row.duration
                start_onset = row.onset
            else:
                run_len += row.duration
        if prev_stage is not None:
            records.append({
                "subject": subj,
                "stage": prev_stage,
                "start_onset": start_onset,
                "duration": run_len,
            })
    ep_df = pd.DataFrame(records)
    ep_df["stage_label"] = stage_labelize(ep_df["stage"])
    fig, ax = plt.subplots(figsize=(8,5))
    sns.boxplot(data=ep_df, x="stage_label", y="duration", ax=ax)
    ax.set_title("Episode duration distribution by stage (micro-epochs aggregated)")
    ax.set_ylabel("Duration (s)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "episode_duration_boxplot.png", dpi=150)
    plt.close(fig)

    # log scale violin
    fig, ax = plt.subplots(figsize=(8,5))
    sns.violinplot(data=ep_df, x="stage_label", y="duration", ax=ax, inner="quart")
    ax.set_title("Episode duration distribution (violin)")
    ax.set_yscale("log")
    ax.set_ylabel("Duration (s, log)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "episode_duration_violin.png", dpi=150)
    plt.close(fig)


def plot_transition_matrix(df_epochs: pd.DataFrame):
    tm = compute_transition_matrix(df_epochs["stage"].tolist())
    tm_lbl = tm.copy()
    tm_lbl.index = [STAGE_MAP.get(i, str(i)) for i in tm.index]
    tm_lbl.columns = [STAGE_MAP.get(i, str(i)) for i in tm.columns]
    fig, ax = plt.subplots(figsize=(6,5))
    sns.heatmap(tm_lbl, annot=True, fmt=".2f", cmap="Blues", ax=ax)
    ax.set_title("Stage transition matrix (probabilities)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "transition_matrix.png", dpi=150)
    plt.close(fig)


def plot_recording_durations(meta_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6,4))
    sns.histplot(meta_df["recording_hours"], bins=30, kde=True, ax=ax)
    ax.set_xlabel("Recording length (hours)")
    ax.set_title("Distribution of recording durations")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "recording_duration_hist.png", dpi=150)
    plt.close(fig)


def main():
    participants = load_participants()
    plot_participant_distribution(participants)

    # Build a combined dataframe of all micro-epochs (4 s windows) across subjects
    all_rows = []
    meta_rows = []
    for ev_path in iter_event_files():
        subject = ev_path.parts[-3]  # sub-XXX
        df = load_events(ev_path)
        # collect meta: total duration
        total_duration = df["duration"].sum()
        meta_rows.append({
            "subject": subject,
            "n_rows": len(df),
            "total_duration_s": total_duration,
            "recording_hours": total_duration / 3600,
        })
        for row in df.itertuples():
            all_rows.append({
                "subject": subject,
                "onset": row.onset,
                "duration": row.duration,
                "stage": row.stage,
            })

    stage_df = pd.DataFrame(all_rows)
    if stage_df.empty:
        print("No event data found.")
        return

    stage_df = stage_df.merge(participants.rename(columns={"participant_id": "subject"}), on="subject", how="left")
    stage_df["stage_label"] = stage_labelize(stage_df["stage"])

    # Summary statistics table
    summary_stats = stage_df.groupby("stage_label").agg(
        count=("stage", "size"),
        total_seconds=("duration", "sum"),
    )
    summary_stats["proportion"] = summary_stats["count"] / summary_stats["count"].sum()
    summary_stats["hours"] = summary_stats["total_seconds"] / 3600
    summary_stats.to_csv(OUTPUT_DIR / "stage_summary_stats.csv")

    # Recording meta
    meta_df = pd.DataFrame(meta_rows)
    meta_df.to_csv(OUTPUT_DIR / "recording_meta.csv", index=False)

    # Plots
    plot_stage_distribution(stage_df)
    plot_episode_lengths(stage_df)
    # Example hypnogram (first subject with most data)
    first_subj = meta_df.sort_values("n_rows", ascending=False).iloc[0]["subject"]
    example_df = stage_df[stage_df["subject"] == first_subj][["onset", "duration", "stage"]].sort_values("onset")
    epochs_30s = aggregate_events(example_df)
    plot_hypnogram(epochs_30s, first_subj)
    plot_transition_matrix(epochs_30s)
    plot_recording_durations(meta_df)

    # Basic textual report
    with open(OUTPUT_DIR / "README.txt", "w") as f:
        f.write("Stage summary (counts, seconds, hours, proportion)\n")
        f.write(summary_stats.to_string())
        f.write("\n\nRecording meta (first 10)\n")
        f.write(meta_df.head(10).to_string())
    print("Exploration complete. Outputs in", OUTPUT_DIR)


if __name__ == "__main__":
    main()
