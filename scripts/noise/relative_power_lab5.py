from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import welch
import pyedflib


LAB_NAME = "lab_5"
TARGET_CHANNELS = ["EEG1", "EEG2"]
BAND_LOW = (0.5, 2.0)
BAND_REF = (0.5, 30.0)

sns.set(style="whitegrid")


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    return here.parents[2]


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
            logging.warning(f"No events.tsv for {edf}; skipping this run (awake-only requested).")
            continue
        runs.append((edf, channels_tsv, ev_path))
    return runs


def read_channels(channels_tsv: Path) -> List[str]:
    ch_df = pd.read_csv(channels_tsv, sep="\t")
    return ch_df["name"].astype(str).tolist()


def _find_channel_indices(reader: pyedflib.EdfReader, wanted: List[str]) -> Dict[str, Optional[int]]:
    labels = [str(x) for x in reader.getSignalLabels()]
    idx_map: Dict[str, Optional[int]] = {}
    lowered = [lab.lower() for lab in labels]
    for w in wanted:
        idx = None
        if w in labels:
            idx = labels.index(w)
        else:
            lw = w.lower()
            if lw in lowered:
                idx = lowered.index(lw)
        idx_map[w] = idx
    return idx_map


def compute_relative_power(x: np.ndarray, fs: float,
                           band: Tuple[float, float],
                           ref_band: Tuple[float, float]) -> float:
    if len(x) < 4:
        return float("nan")
    target_nperseg = int(max(256, min(len(x), fs * 4)))
    try:
        f, pxx = welch(
            x,
            fs=fs,
            nperseg=target_nperseg,
            detrend="constant",
            scaling="density",
            window="hann",
        )
    except ValueError:
        f, pxx = welch(
            x,
            fs=fs,
            nperseg=len(x),
            detrend="constant",
            scaling="density",
            window="hann",
        )

    def band_power(low: float, high: float) -> float:
        mask = (f >= low) & (f <= high)
        if not np.any(mask):
            return 0.0
        trap = getattr(np, "trapezoid", np.trapz)
        return float(trap(pxx[mask], f[mask]))

    p_band = band_power(*band)
    p_ref = band_power(*ref_band)
    if p_ref <= 0:
        return float("nan")
    return p_band / p_ref


def _load_awake_segments(events_tsv: Path) -> List[Tuple[float, float]]:
    ev = pd.read_csv(events_tsv, sep="\t")
    if not {"onset", "duration", "stage"}.issubset(ev.columns):
        raise ValueError(f"events.tsv missing required columns in {events_tsv}")
    awake = ev[ev["stage"] == 1]
    return list(awake[["onset", "duration"]].itertuples(index=False, name=None))


def process_run(edf_path: Path, channels_tsv: Path, events_tsv: Path) -> Dict[str, float]:
    try:
        available_chs = read_channels(channels_tsv)
    except Exception as e:
        logging.warning(f"{edf_path.name}: Could not read channels.tsv ({e}); will rely on EDF labels only.")
        available_chs = []

    missing = [ch for ch in TARGET_CHANNELS if ch not in available_chs]
    if missing and available_chs:
        logging.warning(f"{edf_path.name}: Missing channels from channels.tsv: {missing} — will try EDF labels anyway.")

    rel_powers: Dict[str, float] = {}
    awake_segments = _load_awake_segments(events_tsv)
    if len(awake_segments) == 0:
        logging.warning(f"{edf_path.name}: No Awake epochs found; skipping.")
        for ch in TARGET_CHANNELS:
            rel_powers[ch] = float("nan")
        return rel_powers

    with pyedflib.EdfReader(str(edf_path)) as r:
        idx_map = _find_channel_indices(r, TARGET_CHANNELS)
        for ch in TARGET_CHANNELS:
            idx = idx_map.get(ch)
            if idx is None:
                logging.warning(f"{edf_path.name}: Channel {ch} not found in EDF labels; skipping.")
                rel_powers[ch] = float("nan")
                continue
            fs = float(r.getSampleFrequency(idx))
            sig = r.readSignal(idx)
            x_full = np.asarray(sig, dtype=np.float64)
            slices: List[np.ndarray] = []
            n = len(x_full)
            for onset_sec, dur_sec in awake_segments:
                start = int(round(onset_sec * fs))
                dur = int(round(dur_sec * fs))
                end = start + dur
                if start >= n:
                    continue
                if end > n:
                    end = n
                if end <= start:
                    continue
                slices.append(x_full[start:end])
            if not slices:
                rel_powers[ch] = float("nan")
                continue
            x = np.concatenate(slices, axis=0)
            if not np.isfinite(x).all():
                med = float(np.nanmedian(x))
                x = np.nan_to_num(x, nan=med, posinf=med, neginf=med)
            rel = compute_relative_power(x, fs, BAND_LOW, BAND_REF)
            rel_powers[ch] = float(rel)
    return rel_powers


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
        for edf_path, ch_tsv, ev_tsv in runs:
            rel = process_run(edf_path, ch_tsv, ev_tsv)
            run_id = edf_path.stem
            for ch, val in rel.items():
                records.append({
                    "subject": sub,
                    "run": run_id,
                    "channel": ch,
                    "relative_power_0p5_2_over_0p5_30": val,
                })

    df = pd.DataFrame.from_records(records)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "relative_power_lab5.csv"
    df.to_csv(csv_path, index=False)
    logging.info(f"Wrote {csv_path}")
    return df


def plot_bars(df: pd.DataFrame, out_dir: Path) -> None:
    if df.empty:
        logging.warning("Empty dataframe; skipping plots.")
        return
    agg = (
        df.groupby(["subject", "channel"], as_index=False)["relative_power_0p5_2_over_0p5_30"].mean()
          .rename(columns={"relative_power_0p5_2_over_0p5_30": "relative_power"})
    )

    agg["subject_num"] = agg["subject"].str.extract(r"(\d+)$").astype(int)
    agg = agg.sort_values(["subject_num", "channel"]).reset_index(drop=True)

    for ch in TARGET_CHANNELS:
        sub_df = agg[agg["channel"] == ch]
        if sub_df.empty:
            logging.info(f"No data for channel {ch}; skipping plot.")
            continue
        threshold = 0.2
        plot_df = sub_df.copy()
        plot_df["below_threshold"] = plot_df["relative_power"] < threshold

        plt.figure(figsize=(14, 5))
        ax = sns.barplot(
            data=plot_df,
            x="subject",
            y="relative_power",
            hue="below_threshold",
            palette={True: "#55A868", False: "#C44E52"},
            dodge=False,
        )
        ax.axhline(threshold, color="red", linestyle="--", linewidth=1.5, label=f"Threshold ({threshold})")
        ax.set_title(f"{LAB_NAME}: Relative power {ch} (0.5–2 Hz) / (0.5–30 Hz)")
        ax.set_xlabel("Subject")
        ax.set_ylabel("Relative power")
        plt.xticks(rotation=90)
        handles, labels = ax.get_legend_handles_labels()
        label_map = {"True": "Within limit (below)", "False": "Exceeds limit (above)"}
        labels = [label_map.get(lbl, lbl) for lbl in labels]
        ax.legend(handles, labels, title="Status", loc="upper right")
        plt.tight_layout()
        lab_dir = out_dir / f"{LAB_NAME}_subjects"
        lab_dir.mkdir(parents=True, exist_ok=True)
        out_path = lab_dir / f"{LAB_NAME}_{ch}_relative_power_bar.png"
        plt.savefig(out_path, dpi=200)
        plt.close()
        logging.info(f"Wrote {out_path}")


def plot_per_run(df: pd.DataFrame, out_dir: Path) -> None:
    """Create per-subject, per-channel plots showing all runs individually."""
    if df.empty:
        logging.warning("Empty dataframe; skipping per-run plots.")
        return
    
    threshold = 0.2
    subjects = sorted(df["subject"].unique())
    lab_dir = out_dir / f"{LAB_NAME}_subjects"
    lab_dir.mkdir(parents=True, exist_ok=True)
    
    for sub in subjects:
        for ch in TARGET_CHANNELS:
            sub_ch_df = df[(df["subject"] == sub) & (df["channel"] == ch)].copy()
            if sub_ch_df.empty:
                continue
            
            # Sort by run name for consistent ordering
            sub_ch_df = sub_ch_df.sort_values("run").reset_index(drop=True)
            sub_ch_df["below_threshold"] = sub_ch_df["relative_power_0p5_2_over_0p5_30"] < threshold
            
            plt.figure(figsize=(12, 5))
            ax = sns.barplot(
                data=sub_ch_df,
                x="run",
                y="relative_power_0p5_2_over_0p5_30",
                hue="below_threshold",
                palette={True: "#55A868", False: "#C44E52"},
                dodge=False,
            )
            ax.axhline(threshold, color="red", linestyle="--", linewidth=1.5, label=f"Threshold ({threshold})")
            ax.set_title(f"{LAB_NAME}: {sub} - {ch} Relative power (0.5–2 Hz) / (0.5–30 Hz) per run")
            ax.set_xlabel("Run")
            ax.set_ylabel("Relative power")
            plt.xticks(rotation=45, ha="right")
            handles, labels = ax.get_legend_handles_labels()
            label_map = {"True": "Within limit (below)", "False": "Exceeds limit (above)"}
            labels = [label_map.get(lbl, lbl) for lbl in labels]
            ax.legend(handles, labels, title="Status", loc="upper right")
            plt.tight_layout()
            
            out_path = lab_dir / f"{LAB_NAME}_{sub}_{ch}_relative_power_per_run_bar.png"
            plt.savefig(out_path, dpi=200)
            plt.close()
            logging.info(f"Wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Compute relative band power (0.5–2 Hz)/(0.5–30 Hz) for lab_5 EEG channels (EEG1, EEG2)")
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
    if df.empty:
        logging.warning("No data to plot.")
        return
    plot_bars(df, out_dir)
    plot_per_run(df, out_dir)


if __name__ == "__main__":
    main()
