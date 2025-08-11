"""Data exploration for the ds006366 sleep EEG dataset (mouse sleep EEG).

Generates summary statistics and a range of plots:
1. Participant distribution per lab.
2. Sleep stage distribution (overall + per participant + per lab).
3. Hypnogram example (first subject) at native 4 s epochs.
4. Episode length distribution per stage.
5. Transition matrix between stages.
6. Recording duration distribution.

Note: Mouse sleep is typically scored at high temporal resolution (e.g. 4 s).
We therefore keep the native 4 s epochs (no 30 s aggregation typical of human
sleep scoring) unless the user explicitly requests a larger epoch length.

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
PROCESSED_ROOT = Path("data") / "ds006366_processed"
OUTPUT_DIR = Path("results") / "data_exploration"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Use explicit mapping from task-sleep_events.json
STAGE_MAP = {
    1: "Wake",
    2: "NREM",
    3: "REM",
    4: "Artifact"
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


def aggregate_events(df: pd.DataFrame, epoch_len: int = 4) -> pd.DataFrame:
    """Aggregate native 4 s epochs into larger epochs via majority vote.

    For mouse data the canonical scoring epoch is often already short (4 s).
    With ``epoch_len=4`` (default) this function returns the input sorted by onset
    without aggregation. If ``epoch_len`` is a multiple > 4, consecutive epochs
    are grouped using floor division on (onset - first_onset) and the modal stage
    (majority vote) is assigned. Durations are summed.
    Assumes constant duration per row (observed 4 s). If variable durations are
    present, a duration-weighted vote would be preferable.
    """
    d = df.copy().sort_values("onset")
    if d.empty:
        return d.reset_index(drop=True)
    # If requested epoch equals native resolution -> no aggregation
    native_dur = d["duration"].mode().iloc[0] if not d["duration"].empty else 4
    if epoch_len <= native_dur:
        return d.reset_index(drop=True)
    base = d["onset"].min()
    d["epoch_index"] = ((d["onset"] - base) // epoch_len).astype(int)
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



# No need for heuristic mapping, use STAGE_MAP directly


def stage_labelize(series: pd.Series, mapping: dict[int, str]) -> pd.Series:
    return series.map(lambda x: mapping.get(x, str(x)))


def plot_participant_distribution(participants: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6,4))
    order = participants["lab"].value_counts().index
    # Added hue=x and legend suppression to satisfy upcoming seaborn API (palette without hue deprecated)
    sns.countplot(data=participants, x="lab", hue="lab", order=order, ax=ax, palette="viridis", legend=False)
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
    sns.countplot(data=stage_df, x="stage_label", hue="stage_label", order=order, ax=ax, palette="magma", legend=False)
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


def plot_hypnogram(example_epochs: pd.DataFrame, subject: str, mapping: dict[int, str], epoch_len: int = 4):
    fig, ax = plt.subplots(figsize=(10,3))
    ax.step(example_epochs["onset"] / 3600, example_epochs["stage"], where="post")
    ax.set_xlabel("Hours from lights off")
    ax.set_ylabel("Stage (code)")
    ax.set_title(f"Hypnogram ({epoch_len} s epochs) {subject}")
    stage_codes = sorted([k for k in mapping.keys() if mapping[k] != "Artifact"])
    ax.set_yticks(stage_codes)
    ax.set_yticklabels([mapping[k] for k in stage_codes])
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"hypnogram_{subject}.png", dpi=150)
    plt.close(fig)


def plot_episode_lengths(stage_df: pd.DataFrame, mapping: dict[int, str]):
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
    ep_df["stage_label"] = stage_labelize(ep_df["stage"], mapping)
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


def plot_transition_matrix(df_epochs: pd.DataFrame, mapping: dict[int, str]):
    tm = compute_transition_matrix(df_epochs["stage"].tolist())
    # Remove Artifact from matrix
    valid_codes = [k for k in mapping.keys() if mapping[k] != "Artifact"]
    tm = tm.loc[valid_codes, valid_codes]
    tm_lbl = tm.copy()
    tm_lbl.index = [mapping.get(i, str(i)) for i in tm.index]
    tm_lbl.columns = [mapping.get(i, str(i)) for i in tm.columns]
    fig, ax = plt.subplots(figsize=(6,5))
    sns.heatmap(tm_lbl, annot=True, fmt=".2f", cmap="Blues", ax=ax)
    ax.set_title("Stage transition matrix (probabilities)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "transition_matrix.png", dpi=150)
    plt.close(fig)


def plot_bout_metrics(bout_stats: pd.DataFrame):
    # Mean bout length per stage
    agg = bout_stats.groupby("stage_label").agg(
        mean_bout_length_s=("mean_bout_length_s", "mean"),
        sem_bout_length_s=("mean_bout_length_s", lambda x: x.std(ddof=1)/math.sqrt(len(x))),
        bouts_per_hour=("bouts_per_hour", "mean"),
    ).reset_index()
    # Bout length
    fig, ax = plt.subplots(figsize=(6,4))
    # Add hue for future seaborn compatibility (palette without hue deprecation)
    sns.barplot(data=agg, x="stage_label", y="mean_bout_length_s", hue="stage_label", ax=ax, palette="cubehelix", legend=False)
    ax.set_title("Mean bout length per stage (subject avg)")
    ax.set_ylabel("Seconds")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "mean_bout_length_per_stage.png", dpi=150)
    plt.close(fig)

    # Bouts per hour
    fig, ax = plt.subplots(figsize=(6,4))
    sns.barplot(data=agg, x="stage_label", y="bouts_per_hour", hue="stage_label", ax=ax, palette="viridis", legend=False)
    ax.set_ylabel("Bouts / hour")
    ax.set_title("Fragmentation (bouts per hour)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "bouts_per_hour_per_stage.png", dpi=150)
    plt.close(fig)


def compute_bout_stats(stage_df: pd.DataFrame, mapping: dict[int, str]) -> pd.DataFrame:
    records = []
    for subject, df_sub in stage_df.groupby("subject"):
        df_sub = df_sub.sort_values("onset")
        total_hours = df_sub["duration"].sum() / 3600
        # Identify episodes
        episodes = []
        prev_stage = None
        run_len = 0
        for row in df_sub.itertuples():
            if row.stage != prev_stage:
                if prev_stage is not None:
                    episodes.append((prev_stage, run_len))
                prev_stage = row.stage
                run_len = row.duration
            else:
                run_len += row.duration
        if prev_stage is not None:
            episodes.append((prev_stage, run_len))
        # per stage metrics
        by_stage = {}
        for st, dur in episodes:
            by_stage.setdefault(st, []).append(dur)
        for st, durs in by_stage.items():
            arr = np.array(durs)
            records.append({
                "subject": subject,
                "stage": st,
                "stage_label": mapping.get(st, str(st)),
                "mean_bout_length_s": arr.mean(),
                "median_bout_length_s": np.median(arr),
                "bouts_per_hour": len(arr)/total_hours if total_hours>0 else np.nan,
            })
    return pd.DataFrame(records)


def plot_bout_size_distribution(stage_df: pd.DataFrame, mapping: dict[int, str]):
    # Episode extraction
    episodes = []
    for subj, df_sub in stage_df.groupby("subject"):
        df_sub = df_sub.sort_values("onset")
        prev_stage = None
        run_len = 0
        for row in df_sub.itertuples():
            if row.stage != prev_stage:
                if prev_stage is not None:
                    episodes.append({"subject": subj, "stage": prev_stage, "duration": run_len})
                prev_stage = row.stage
                run_len = row.duration
            else:
                run_len += row.duration
        if prev_stage is not None:
            episodes.append({"subject": subj, "stage": prev_stage, "duration": run_len})
    ep_df = pd.DataFrame(episodes)
    if ep_df.empty:
        return
    bins = [0,4,32,60,300,1e9]
    labels = ["4s","4-32s","32-60s","60-300s","300s+"]
    ep_df["bin"] = pd.cut(ep_df["duration"], bins=bins, labels=labels, right=False)
    ep_df["stage_label"] = ep_df["stage"].map(lambda x: mapping.get(x, str(x)))
    # Compute proportions per stage using an explicit intermediate to avoid reset_index duplication error
    counts = ep_df.groupby(["stage_label","bin"]).size().reset_index(name="count")
    counts["proportion"] = counts.groupby("stage_label")["count"].transform(lambda x: x / x.sum())
    prop = counts.drop(columns=["count"])
    fig, ax = plt.subplots(figsize=(8,4))
    # stacked bar
    stages = prop["stage_label"].unique()
    bottom = np.zeros(len(stages))
    stage_index = {st:i for i, st in enumerate(stages)}
    for b in labels:
        vals = [prop[(prop.stage_label==st)&(prop.bin==b)]["proportion"].values[0] if not prop[(prop.stage_label==st)&(prop.bin==b)].empty else 0 for st in stages]
        ax.bar(stages, vals, bottom=bottom, label=b)
        bottom += vals
    ax.set_ylabel("Proportion of episodes")
    ax.set_title("Bout size distribution per stage")
    ax.legend(title="Bout length bin", bbox_to_anchor=(1.05,1), loc="upper left")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "bout_size_distribution.png", dpi=150)
    plt.close(fig)


def plot_recording_durations(meta_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6,4))
    sns.histplot(meta_df["recording_hours"], bins=30, kde=True, ax=ax)
    ax.set_xlabel("Recording length (hours)")
    ax.set_title("Distribution of recording durations")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "recording_duration_hist.png", dpi=150)
    plt.close(fig)


# ------------------ Simple processed data amplitude analysis (single run) ------------------ #
def analyze_single_run_amplitude(subject: str = "sub-087", run: str = "1", channel: str = "EEG1") -> dict:
    """Load one processed run (e.g. data/ds006366_processed/sub-087/1/EEG1.npy) and compute epoch-level amplitude statistics.

    Returns a dictionary with arrays and summary stats.
    """
    run_dir = PROCESSED_ROOT / subject / run
    eeg_path = run_dir / f"{channel}.npy"
    labels_path = run_dir / "labels.npy"
    if not eeg_path.exists():
        raise FileNotFoundError(f"EEG file not found: {eeg_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"labels.npy not found: {labels_path}")
    eeg = np.load(eeg_path)
    labels = np.load(labels_path)
    n_epochs = len(labels)
    if eeg.ndim == 2 and eeg.shape[0] < 10:  # channels x samples
        eeg_sig = eeg[0]
    else:
        eeg_sig = eeg
    total_samples = eeg_sig.shape[-1]
    samples_per_epoch = total_samples // n_epochs
    if samples_per_epoch * n_epochs != total_samples:
        raise ValueError("Samples do not divide evenly by number of epochs")
    epochs = eeg_sig.reshape(n_epochs, samples_per_epoch)
    epoch_std = epochs.std(axis=1)  # uV
    epoch_ptp = (epochs.max(axis=1) - epochs.min(axis=1))  # peak-to-peak
    stats = {
        "subject": subject,
        "run": run,
        "channel": channel,
        "n_epochs": n_epochs,
        "samples_per_epoch": samples_per_epoch,
        "mean_std_uV": float(epoch_std.mean()),
        "median_std_uV": float(np.median(epoch_std)),
        "std_of_std_uV": float(epoch_std.std(ddof=1)),
        "min_std_uV": float(epoch_std.min()),
        "max_std_uV": float(epoch_std.max()),
        "p25_std_uV": float(np.percentile(epoch_std, 25)),
        "p75_std_uV": float(np.percentile(epoch_std, 75)),
        "mean_ptp_uV": float(epoch_ptp.mean()),
        "median_ptp_uV": float(np.median(epoch_ptp)),
    }
    # Plot epoch std over epochs
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(np.arange(n_epochs), epoch_std, lw=0.8)
    ax.set_xlabel("Epoch index")
    ax.set_ylabel("Amplitude (std, uV)")
    ax.set_title(f"Epoch amplitude (std) {subject} {run} {channel}")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{subject}_{run}_{channel}_epoch_std.png", dpi=150)
    plt.close(fig)
    # Histogram of epoch std
    fig, ax = plt.subplots(figsize=(6,4))
    sns.histplot(epoch_std, bins=30, kde=True, ax=ax)
    ax.set_xlabel("Epoch std (uV)")
    ax.set_title(f"Distribution of epoch std {subject} {run} {channel}")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{subject}_{run}_{channel}_epoch_std_hist.png", dpi=150)
    plt.close(fig)
    # Save stats
    with open(OUTPUT_DIR / f"{subject}_{run}_{channel}_amplitude_stats.txt", "w") as f:
        for k,v in stats.items():
            f.write(f"{k}: {v}\n")
        f.write("Recommended base_amplitude (mean std uV): %.3f\n" % stats["mean_std_uV"])
    return {"epoch_std": epoch_std, "epoch_ptp": epoch_ptp, "stats": stats}



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
    # Use explicit mapping
    mapping = STAGE_MAP
    # Remove Artifact from all stats/plots
    valid_mask = stage_df["stage"].isin([k for k,v in mapping.items() if v != "Artifact"])
    stage_df = stage_df[valid_mask].copy()
    stage_df["stage_label"] = stage_labelize(stage_df["stage"], mapping)

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
    plot_episode_lengths(stage_df, mapping)
    # Example hypnogram (first subject with most data) at native 4 s epochs
    first_subj = meta_df.sort_values("n_rows", ascending=False).iloc[0]["subject"]
    example_df = stage_df[stage_df["subject"] == first_subj][["onset", "duration", "stage"]].sort_values("onset")
    epochs_4s = aggregate_events(example_df, epoch_len=4)  # returns unaggregated
    plot_hypnogram(epochs_4s, first_subj, mapping, epoch_len=4)
    plot_transition_matrix(epochs_4s, mapping)
    plot_recording_durations(meta_df)

    # Bout metrics & fragmentation
    bout_stats = compute_bout_stats(stage_df, mapping)
    if not bout_stats.empty:
        bout_stats.to_csv(OUTPUT_DIR / "bout_stats.csv", index=False)
        plot_bout_metrics(bout_stats)
        plot_bout_size_distribution(stage_df, mapping)

    # ---- Processed data: single-run amplitude analysis (sub-087 run 1) ---- #
    try:
        amplitude_result = analyze_single_run_amplitude(subject="sub-087", run="1", channel="EEG1")
        print("Amplitude stats (uV):", amplitude_result["stats"])
    except Exception as e:
        print("Amplitude analysis skipped:", e)

    # Basic textual report
    with open(OUTPUT_DIR / "README.txt", "w") as f:
        f.write("Stage summary (counts, seconds, hours, proportion)\n")
        f.write(summary_stats.to_string())
        f.write("\n\nStage mapping (numeric->label):\n")
        for k,v in mapping.items():
            f.write(f"  {k} -> {v}\n")
        if 'bout_stats' in locals():
            f.write("\nBout statistics (first 10 rows):\n")
            f.write(bout_stats.head(10).to_string())
        f.write("\n\nRecording meta (first 10)\n")
        f.write(meta_df.head(10).to_string())
    print("Exploration complete. Outputs in", OUTPUT_DIR)


if __name__ == "__main__":
    main()
