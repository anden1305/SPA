from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pyedflib


LAB_NAME = "lab_2"
EMG_CHANNEL = "EMG"

sns.set(style="whitegrid")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_data_root() -> Path:
    return _repo_root() / "data" / "ds006366"


def default_out_dir() -> Path:
    return _repo_root() / "results" / "noise"


def read_lab_subjects(participants_tsv: Path, lab: str,
                      subjects: Optional[List[str]] = None,
                      max_subjects: Optional[int] = None) -> List[str]:
    df = pd.read_csv(participants_tsv, sep="\t")
    df = df[df["lab"] == lab]
    if subjects:
        df = df[df["participant_id"].isin(subjects)]
    subs = df["participant_id"].tolist()
    if max_subjects is not None:
        subs = subs[:max_subjects]
    return subs


def list_runs_for_subject(sub_dir: Path, max_runs: Optional[int] = None) -> List[Tuple[Path, Path, Path]]:
    eeg_dir = sub_dir / "eeg"
    if not eeg_dir.exists():
        return []
    edfs = sorted(eeg_dir.glob("*_eeg.edf"))
    if max_runs is not None:
        edfs = edfs[:max_runs]
    channels_candidates = sorted(eeg_dir.glob("*_channels.tsv"))
    channels_tsv = channels_candidates[0] if channels_candidates else None
    runs: List[Tuple[Path, Path, Path]] = []
    for edf in edfs:
        if channels_tsv is None:
            logging.warning(f"No channels.tsv for {edf}")
            continue
        ev_path = edf.with_name(edf.name.replace("_eeg.edf", "_events.tsv"))
        if not ev_path.exists():
            cand = list(eeg_dir.glob(edf.stem.replace("_eeg", "") + "*_events.tsv"))
            ev_path = cand[0] if cand else ev_path
        if not ev_path.exists():
            logging.warning(f"No events.tsv for {edf}; skipping run.")
            continue
        runs.append((edf, channels_tsv, ev_path))
    return runs


def _find_emg_index(reader: pyedflib.EdfReader) -> Optional[int]:
    labels = [str(x) for x in reader.getSignalLabels()]
    lowered = [lab.lower() for lab in labels]
    if EMG_CHANNEL in labels:
        return labels.index(EMG_CHANNEL)
    if EMG_CHANNEL.lower() in lowered:
        return lowered.index(EMG_CHANNEL.lower())
    for i, lab in enumerate(lowered):
        if "emg" in lab:
            return i
    return None


def _load_segments(events_tsv: Path, stage_value: int) -> List[Tuple[float, float]]:
    ev = pd.read_csv(events_tsv, sep="\t")
    if not {"onset", "duration", "stage"}.issubset(ev.columns):
        raise ValueError(f"events.tsv missing required columns in {events_tsv}")
    segs = ev[ev["stage"] == stage_value]
    return list(segs[["onset", "duration"]].itertuples(index=False, name=None))


def _concat_segments(x_full: np.ndarray, fs: float, segments: List[Tuple[float, float]]) -> np.ndarray:
    parts: List[np.ndarray] = []
    n = len(x_full)
    for onset_sec, dur_sec in segments:
        start = int(round(onset_sec * fs))
        dur = int(round(dur_sec * fs))
        end = start + dur
        if start >= n:
            continue
        if end > n:
            end = n
        if end <= start:
            continue
        parts.append(x_full[start:end])
    if not parts:
        return np.array([], dtype=x_full.dtype)
    return np.concatenate(parts, axis=0)


def _rms(x: np.ndarray) -> float:
    if x.size == 0:
        return float("nan")
    if not np.isfinite(x).all():
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))


def process_run(edf_path: Path, events_tsv: Path) -> float:
    awake_segments = _load_segments(events_tsv, stage_value=1)  # 1=Wake
    nrem_segments = _load_segments(events_tsv, stage_value=2)   # 2=NREM
    if len(awake_segments) == 0 or len(nrem_segments) == 0:
        logging.warning(f"{edf_path.name}: Missing Awake or NREM epochs; returning NaN")
        return float("nan")

    with pyedflib.EdfReader(str(edf_path)) as r:
        idx = _find_emg_index(r)
        if idx is None:
            logging.warning(f"{edf_path.name}: EMG channel not found; returning NaN")
            return float("nan")
        fs = float(r.getSampleFrequency(idx))
        sig = r.readSignal(idx)
        x_full = np.asarray(sig, dtype=np.float64)

        x_awake = _concat_segments(x_full, fs, awake_segments)
        x_nrem = _concat_segments(x_full, fs, nrem_segments)

        rms_awake = _rms(x_awake)
        rms_nrem = _rms(x_nrem)

        if not np.isfinite(rms_awake) or not np.isfinite(rms_nrem) or rms_nrem == 0:
            return float("nan")
        return rms_awake / rms_nrem


def run_analysis(data_root: Path, out_dir: Path,
                 subjects: Optional[List[str]] = None,
                 max_subjects: Optional[int] = None,
                 max_runs: Optional[int] = None) -> pd.DataFrame:
    participants_tsv = data_root / "participants.tsv"
    subs = read_lab_subjects(participants_tsv, LAB_NAME, subjects=subjects, max_subjects=max_subjects)
    if not subs:
        raise RuntimeError(f"No participants found for {LAB_NAME} in {participants_tsv}")

    records = []
    for sub in subs:
        sub_dir = data_root / sub
        runs = list_runs_for_subject(sub_dir, max_runs=max_runs)
        if not runs:
            logging.warning(f"{sub}: no runs found")
            continue
        for edf_path, _ch_tsv, ev_tsv in runs:
            ratio = process_run(edf_path, ev_tsv)
            records.append({
                "subject": sub,
                "run": edf_path.stem,
                "emg_rms_awake_over_nrem": ratio,
            })

    df = pd.DataFrame.from_records(records)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "emg_rms_awake_over_nrem_lab2.csv"
    df.to_csv(csv_path, index=False)
    logging.info(f"Wrote {csv_path}")
    return df


def plot_bars(df: pd.DataFrame, out_dir: Path) -> None:
    if df.empty:
        logging.warning("Empty dataframe; skipping plot.")
        return
    agg = df.groupby(["subject"], as_index=False)["emg_rms_awake_over_nrem"].mean()
    agg["subject_num"] = agg["subject"].str.extract(r"(\d+)$").astype(int)
    agg = agg.sort_values(["subject_num"]).reset_index(drop=True)

    threshold = 2.0
    plot_df = agg.copy()
    plot_df["below_threshold"] = plot_df["emg_rms_awake_over_nrem"] < threshold

    plt.figure(figsize=(14, 5))
    ax = sns.barplot(
        data=plot_df,
        x="subject",
        y="emg_rms_awake_over_nrem",
        hue="below_threshold",
        palette={False: "#55A868", True: "#C44E52"},
        dodge=False,
    )
    ax.axhline(threshold, color="red", linestyle="--", linewidth=1.5, label=f"Threshold ({threshold})")
    ax.set_title(f"{LAB_NAME}: EMG RMS(Awake) / RMS(NREM)")
    ax.set_xlabel("Subject")
    ax.set_ylabel("RMS ratio (Awake / NREM)")
    plt.xticks(rotation=90)
    handles, labels = ax.get_legend_handles_labels()
    label_map = {"True": "Below threshold", "False": "Above threshold"}
    labels = [label_map.get(lbl, lbl) for lbl in labels]
    ax.legend(handles, labels, title="Status", loc="upper right")
    plt.tight_layout()
    lab_dir = out_dir / f"{LAB_NAME}_subjects"
    lab_dir.mkdir(parents=True, exist_ok=True)
    out_path = lab_dir / f"{LAB_NAME}_EMG_rms_awake_over_nrem_bar.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    logging.info(f"Wrote {out_path}")


def plot_per_run(df: pd.DataFrame, out_dir: Path) -> None:
    """Create per-subject plots showing all runs individually."""
    if df.empty:
        logging.warning("Empty dataframe; skipping per-run plots.")
        return
    
    threshold = 2.0
    subjects = sorted(df["subject"].unique())
    
    for sub in subjects:
        sub_df = df[df["subject"] == sub].copy()
        if sub_df.empty:
            continue
        
        # Sort by run name for consistent ordering
        sub_df = sub_df.sort_values("run").reset_index(drop=True)
        sub_df["below_threshold"] = sub_df["emg_rms_awake_over_nrem"] < threshold
        
        plt.figure(figsize=(12, 5))
        ax = sns.barplot(
            data=sub_df,
            x="run",
            y="emg_rms_awake_over_nrem",
            hue="below_threshold",
            palette={False: "#55A868", True: "#C44E52"},
            dodge=False,
        )
        ax.axhline(threshold, color="red", linestyle="--", linewidth=1.5, label=f"Threshold ({threshold})")
        ax.set_title(f"{LAB_NAME}: {sub} - EMG RMS(Awake) / RMS(NREM) per run")
        ax.set_xlabel("Run")
        ax.set_ylabel("RMS ratio (Awake / NREM)")
        plt.xticks(rotation=45, ha="right")
        handles, labels = ax.get_legend_handles_labels()
        label_map = {"True": "Below threshold", "False": "Above threshold"}
        labels = [label_map.get(lbl, lbl) for lbl in labels]
        ax.legend(handles, labels, title="Status", loc="upper right")
        plt.tight_layout()
        
        lab_dir = out_dir / f"{LAB_NAME}_subjects"
        lab_dir.mkdir(parents=True, exist_ok=True)
        out_path = lab_dir / f"{LAB_NAME}_{sub}_EMG_rms_per_run_bar.png"
        plt.savefig(out_path, dpi=200)
        plt.close()
        logging.info(f"Wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Compute EMG RMS(Awake)/RMS(NREM) for lab_2 subjects and plot per-subject bars")
    parser.add_argument("--data-root", type=str, default=None,
                        help="Path to ds006366 dataset root (default: <repo>/data/ds006366)")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Directory to write results (default: <repo>/results/noise)")
    parser.add_argument("--subjects", type=str, default=None,
                        help="Comma-separated list of subject IDs to include (e.g., sub-038,sub-039)")
    parser.add_argument("--max-subjects", type=int, default=None,
                        help="Limit number of subjects (useful for quick checks)")
    parser.add_argument("--max-runs", type=int, default=None,
                        help="Limit number of runs per subject (useful for quick checks)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    data_root = Path(args.data_root) if args.data_root else default_data_root()
    out_dir = Path(args.out_dir) if args.out_dir else default_out_dir()
    subjects = [s.strip() for s in args.subjects.split(",")] if args.subjects else None

    logging.info(f"Data root: {data_root}")
    logging.info(f"Output dir: {out_dir}")

    df = run_analysis(
        data_root,
        out_dir,
        subjects=subjects,
        max_subjects=args.max_subjects,
        max_runs=args.max_runs,
    )
    plot_bars(df, out_dir)
    plot_per_run(df, out_dir)


if __name__ == "__main__":
    main()
