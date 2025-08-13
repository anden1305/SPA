"""General dataset overview: labs, participants, runs, signals.
Generates summary tables and five core plots.
"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from helpers import load_metadata, load_participant_data, ensure_output_directory

plt.style.use('seaborn-v0_8')


# ------------------------------------------------------------
# Helper: Plot raw EEG excerpt (optionally with hypnogram labels)
# ------------------------------------------------------------
def _plot_raw_eeg_excerpt(
    data_dict,
    sampling_rate,
    out_dir,
    channel="EEG1",
    duration_sec=80,
    start_sec=0,
    stage_colors=None,
    stage_name_map=None,
    file_prefix="raw_eeg",
    subject=None,
    run=None,
    time=None
):
    """Create two plots: raw signal and (if labels available) labeled signal.

    data_dict: output of load_participant_data(pid, run)
    sampling_rate: Hz
    channel: which EEG key to look for (fallback: first EEG* present)
    duration_sec: length of excerpt
    start_sec: starting second within the run
    stage_colors: mapping stage_name -> RGBA/hex
    stage_name_map: mapping raw label values -> stage_name
    file_prefix: base filename (without extension). If left as default 'raw_eeg' and
                 subject/run/time provided, an augmented prefix will be auto-built.
    subject: optional subject identifier string (e.g. 'sub-032') used in titles / filenames
    run: optional run identifier (int or str)
    time: optional reference time label (e.g. absolute clock / lights-off seconds) for annotation
    """
    if channel not in data_dict:
        # Fallback: pick the first EEG-like channel
        eeg_candidates = [k for k in data_dict if k.upper().startswith("EEG")]
        # Deterministic order (in case dict order varies across environments)
        if eeg_candidates:
            channel = sorted(eeg_candidates, key=str.lower)[0]
        else:
            return  # nothing to plot

    signal = data_dict[channel]
    n_total = len(signal)
    start_idx = int(start_sec * sampling_rate)
    end_idx = min(start_idx + int(duration_sec * sampling_rate), n_total)
    if start_idx >= end_idx:
        return
    excerpt = signal[start_idx:end_idx]
    t = (np.arange(len(excerpt)) / sampling_rate) + start_sec

    out_dir.mkdir(exist_ok=True, parents=True)

    # Derive effective file prefix (non-breaking: only auto-augment when user did not pass custom)
    effective_prefix = file_prefix
    if file_prefix == "raw_eeg" and any(v is not None for v in (subject, run, time)):
        parts = [file_prefix]
        if subject:
            parts.append(str(subject))
        if run is not None:
            parts.append(f"run-{run}")
        if time is not None:
            parts.append(f"t{int(time)}s" if isinstance(time, (int, float)) else str(time))
        effective_prefix = "_".join(parts)

    # Helper to build title suffix
    meta_bits = []
    if subject:
        meta_bits.append(f"sub={subject}")
    if run is not None:
        meta_bits.append(f"run={run}")
    if time is not None:
        meta_bits.append(f"t0={time}s" if isinstance(time, (int, float)) else f"time={time}")
    title_suffix = f" [{' ,'.join(meta_bits)}]" if meta_bits else ""

    # 1) Plain raw plot
    plt.figure(figsize=(16,3))
    plt.plot(t, excerpt, color='black', linewidth=0.8)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (µV)')
    plt.title(f'Raw EEG Excerpt ({int(duration_sec)}s) - {channel}{title_suffix}')
    plt.tight_layout()
    plt.savefig(out_dir / f'{effective_prefix}_excerpt.png', dpi=160)
    plt.close()

    # 2) Labeled plot (if labels exist)
    if 'labels' not in data_dict:
        return
    labels = np.array(data_dict['labels'])
    if labels.size == 0:
        return

    # Attempt to align labels to samples. Common cases:
    # a) one label per sample (len==n_total)
    # b) one label per epoch (30s, 20s, 10s etc.)
    if labels.shape[0] == n_total:
        epoch_len_samples = 1
    else:
        # infer by integer division
        ratios = n_total / labels.shape[0]
        epoch_len_samples = int(round(ratios)) if ratios >= 1 else 1
    # Build stage intervals covering the excerpt
    excerpt_labels = []
    # Determine label index range overlapping excerpt
    first_label_idx = start_idx // epoch_len_samples
    last_label_idx = (end_idx - 1) // epoch_len_samples
    for li in range(first_label_idx, last_label_idx + 1):
        if li < 0 or li >= labels.shape[0]:
            continue
        raw_val = labels[li]
        # Map value to standardized stage name
        if stage_name_map:
            stage_name = stage_name_map.get(raw_val, str(raw_val))
        else:
            # Try some defaults
            default_map = {
                0: 'AWAKE', 1: 'NREM', 2: 'REM', 3: 'NREM', 4: 'NREM',
                'W': 'AWAKE', 'R': 'REM', 'N1': 'NREM', 'N2': 'NREM', 'N3': 'NREM'
            }
            stage_name = default_map.get(raw_val, str(raw_val))
        seg_start = li * epoch_len_samples / sampling_rate
        seg_end = (li + 1) * epoch_len_samples / sampling_rate
        # Clip to excerpt window
        seg_start_clipped = max(seg_start, start_sec)
        seg_end_clipped = min(seg_end, start_sec + duration_sec)
        if seg_end_clipped <= seg_start_clipped:
            continue
        excerpt_labels.append((stage_name, seg_start_clipped, seg_end_clipped))

    if not excerpt_labels:
        return

    if stage_colors is None:
        stage_colors = {
            'AWAKE': '#F8B5C7',
            'NREM': '#B6D8C0',
            'REM': '#B3D9F2'
        }

    plt.figure(figsize=(16,3))
    plt.plot(t, excerpt, color='black', linewidth=0.8)
    # Background blocks
    for stage_name, s0, s1 in excerpt_labels:
        color = stage_colors.get(stage_name, '#E0E0E0')
        plt.axvspan(s0, s1, color=color, alpha=0.35, linewidth=0)
    # Vertical dotted lines every 4s (like example) within window
    span_start = start_sec
    span_end = start_sec + duration_sec
    for v in np.arange(span_start, span_end, 4):
        plt.axvline(v, color='k', linestyle=':', linewidth=0.4, alpha=0.6)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (µV)')
    plt.title(f'Raw EEG Labeled Excerpt ({int(duration_sec)}s) - {channel}{title_suffix}')
    # Legend (unique order-preserving)
    seen = []
    for stage_name, _, _ in excerpt_labels:
        if stage_name not in seen:
            seen.append(stage_name)
    handles = [plt.Line2D([0],[0], color='white', marker='s', markersize=10,
                markerfacecolor=stage_colors.get(st, '#E0E0E0'), label=st) for st in seen]
    if handles:
        plt.legend(handles=handles, title='Stage', frameon=True, loc='upper right')
    plt.tight_layout()
    plt.savefig(out_dir / f'{effective_prefix}_labeled_excerpt.png', dpi=160)
    plt.close()


_CACHED_EPOCH_SEC = None

def _infer_epoch_sec(meta, sampling_rate, data_dir="data/ds006366_processed"):
    """Infer epoch duration in seconds once (assumes constant across dataset).
    Loads labels and the first available EEG channel for the first row only."""
    global _CACHED_EPOCH_SEC
    if _CACHED_EPOCH_SEC is not None:
        return _CACHED_EPOCH_SEC
    for _, row in meta.iterrows():
        pid, run_id = row['participant_id'], int(row['run'])
        base = Path(data_dir) / pid / str(run_id)
        labels_path = base / 'labels.npy'
        if not labels_path.exists():
            continue
        try:
            labels = np.load(labels_path, mmap_mode='r')
        except Exception:
            continue
        # Try EEG1..4
        eeg_len = None
        for ch in ['EEG1','EEG2','EEG3','EEG4']:
            p = base / f'{ch}.npy'
            if p.exists():
                try:
                    eeg_len = np.load(p, mmap_mode='r').shape[0]
                    break
                except Exception:
                    pass
        if eeg_len is None:
            continue
        if labels.shape[0] == 0:
            continue
        # Epoch length in samples (assumed integer)
        samples_per_epoch = max(1, int(round(eeg_len / labels.shape[0])))
        _CACHED_EPOCH_SEC = samples_per_epoch / sampling_rate
        return _CACHED_EPOCH_SEC
    # Fallback assume 30s epochs
    _CACHED_EPOCH_SEC = 30.0
    return _CACHED_EPOCH_SEC

def _find_high_transition_excerpt(meta, sampling_rate, duration_sec, max_runs_scan=500, min_transitions=4, data_dir="data/ds006366_processed"):
    """Very fast search using only labels.npy for each run.

    Returns (participant_id, run, start_sec, n_transitions) for first qualifying window.
    The start_sec is computed after epoch length inference (one EEG load total).
    """
    epoch_sec = _infer_epoch_sec(meta, sampling_rate, data_dir=data_dir)
    window_epochs = max(2, int(round(duration_sec / epoch_sec)))  # need at least 2 epochs
    scan_meta = meta.head(max_runs_scan)
    for _, row in scan_meta.iterrows():
        pid, run_id = row['participant_id'], int(row['run'])
        labels_path = Path(data_dir) / pid / str(run_id) / 'labels.npy'
        if not labels_path.exists():
            continue
        try:
            labels = np.load(labels_path, mmap_mode='r')  # memory-map for speed/low RAM
        except Exception:
            continue
        n_epochs = labels.shape[0]
        if n_epochs < 2 or window_epochs > n_epochs:
            continue
        # Precompute transitions between epochs
        diffs = (labels[1:] != labels[:-1]).astype(np.uint8)
        if diffs.sum() < min_transitions:
            continue
        k = window_epochs - 1  # number of diffs inside a window
        if diffs.size < k:
            continue
        # Cumulative sum for O(1) window transition counts
        csum = np.concatenate([[0], np.cumsum(diffs)])
        # Iterate windows; early break when found
        # vectorized check in blocks for speed
        counts = csum[k:] - csum[:-k]
        hit_idx = np.argmax(counts >= min_transitions) if np.any(counts >= min_transitions) else -1
        if hit_idx >= 0:
            start_epoch = hit_idx
            start_sec = start_epoch * epoch_sec
            return (pid, run_id, float(start_sec), int(counts[hit_idx]))
    return (None, None, None, 0)


def analyze_dataset_overview(output_dir="results/data_exploration", sampling_rate=128, max_participants_scan=500):
    print("="*60)
    print("DATASET OVERVIEW")
    print("="*60)
    out_base = Path(ensure_output_directory(output_dir)) / "dataset_overview"
    out_base.mkdir(exist_ok=True)

    meta = load_metadata()
    meta['run'] = meta['run'].astype(int)

    # 1. Basic counts
    n_labs = meta['lab'].nunique()
    n_participants = meta['participant_id'].nunique()
    runs_per_participant = meta.groupby('participant_id')['run'].nunique()
    total_runs = len(meta)
    participants_per_lab = meta.groupby('lab')['participant_id'].nunique().sort_values(ascending=False)
    runs_per_lab = meta.groupby('lab')['run'].count().sort_values(ascending=False)

    basic_df = pd.DataFrame({
        'metric': [
            'n_labs','n_participants','total_runs','median_runs_per_participant','mean_runs_per_participant'
        ],
        'value': [
            n_labs, n_participants, total_runs, runs_per_participant.median(), runs_per_participant.mean()
        ]
    })
    basic_df.to_csv(out_base / 'basic_counts.csv', index=False)

    participants_per_lab.to_csv(out_base / 'participants_per_lab.csv', header=['n_participants'])
    runs_per_lab.to_csv(out_base / 'runs_per_lab.csv', header=['n_runs'])

    # 2. Scan subset for signal availability (avoid loading everything)
    sampled_meta = meta.groupby('participant_id').head(1).head(max_participants_scan)
    signal_presence = []
    unique_signals = set()
    for _, row in sampled_meta.iterrows():
        pid, run_id = row['participant_id'], row['run']
        try:
            data = load_participant_data(pid, run_id)
        except Exception:
            continue
        present = [k for k in data.keys() if k != 'labels']
        for sig in present:
            unique_signals.add(sig)
        signal_presence.append({'participant_id': pid, 'run': run_id, **{sig: (sig in present) for sig in ['EEG1','EEG2','EEG3','EEG4','EMG']}})

    signal_df = pd.DataFrame(signal_presence)
    if not signal_df.empty:
        signal_df.to_csv(out_base / 'signal_presence_sample.csv', index=False)

    signal_counts = {sig: (signal_df[sig].sum() if sig in signal_df else 0) for sig in ['EEG1','EEG2','EEG3','EEG4','EMG']}
    signal_counts_df = pd.DataFrame([signal_counts])
    signal_counts_df.to_csv(out_base / 'signal_counts_sample.csv', index=False)

    # Derived table summarizing participants per #runs bucket
    # Distribution of number of runs per participant
    run_bucket = runs_per_participant.value_counts().sort_index()
    run_bucket_df = run_bucket.reset_index()
    run_bucket_df.columns = ['n_runs','n_participants']
    run_bucket_df.to_csv(out_base / 'participants_by_run_count.csv', index=False)

    # ---------------- Plots (5) ----------------
    sns.set_palette('viridis')

    # Plot 1: Participants per lab
    plt.figure(figsize=(8,4))
    participants_per_lab.plot(kind='bar')
    plt.ylabel('Participants')
    plt.title('Participants per Lab')
    plt.tight_layout()
    plt.savefig(out_base / 'participants_per_lab.png', dpi=160)
    plt.close()

    # Plot 2: Runs per lab
    plt.figure(figsize=(8,4))
    runs_per_lab.plot(kind='bar', color='teal')
    plt.ylabel('Runs')
    plt.title('Runs per Lab')
    plt.tight_layout()
    plt.savefig(out_base / 'runs_per_lab.png', dpi=160)
    plt.close()

    # Plot 3: Distribution of runs per participant
    plt.figure(figsize=(6,4))
    sns.histplot(runs_per_participant, bins=range(1, runs_per_participant.max()+2), kde=False, color='#4ECDC4', edgecolor='black')
    plt.xlabel('Runs per Participant')
    plt.ylabel('Count of Participants')
    plt.title('Distribution of Runs per Participant')
    plt.tight_layout()
    plt.savefig(out_base / 'runs_per_participant_distribution.png', dpi=160)
    plt.close()

    # Plot 4: Signal availability heatmap (sample)
    if not signal_df.empty:
        plt.figure(figsize=(6,4))
        heat_data = signal_df[['EEG1','EEG2','EEG3','EEG4','EMG']].mean().to_frame(name='availability')
        sns.heatmap(heat_data, annot=True, fmt='.2f', cmap='YlGnBu', cbar=False)
        plt.title('Signal Availability Fraction (Sample)')
        plt.tight_layout()
        plt.savefig(out_base / 'signal_availability_heatmap.png', dpi=160)
        plt.close()

    # Plot 5: Participants by number of runs
    if not run_bucket_df.empty:
        plt.figure(figsize=(6,4))
        sns.barplot(data=run_bucket_df, x='n_runs', y='n_participants', color='#FF6B6B')
        plt.xlabel('Number of Runs')
        plt.ylabel('Participants')
        plt.title('Participants by Run Count')
        for i, row in run_bucket_df.iterrows():
            plt.text(i, row['n_participants'], int(row['n_participants']), ha='center', va='bottom', fontsize=8)
        plt.tight_layout()
        plt.savefig(out_base / 'participants_by_run_count.png', dpi=160)
        plt.close()
    else:
        print("Warning: run_bucket_df is empty; skipping participants_by_run_count plot.")

    # Combined overview summary markdown
    overview_lines = [
        f"Number of labs: {n_labs}",
        f"Number of participants: {n_participants}",
        f"Total runs: {total_runs}",
        f"Median runs per participant: {runs_per_participant.median():.1f}",
        f"Mean runs per participant: {runs_per_participant.mean():.2f}",
        f"Signals (sampled): {', '.join(sorted(unique_signals)) if unique_signals else 'N/A'}"
    ]
    (out_base / 'overview_summary.txt').write_text("\n".join(overview_lines))
    print("\n".join(overview_lines))
    print(f"Dataset overview tables & plots written to {out_base}")

    # ------------------------------------------------------------
    # Raw EEG excerpt plots (similar style to synthetic examples)
    # ------------------------------------------------------------
    try:
        sample_row = meta.iloc[0]
        pid, run_id = sample_row['participant_id'], int(sample_row['run'])
        data = load_participant_data(pid, run_id)
        eeg_out = out_base / 'raw_eeg'
        _plot_raw_eeg_excerpt(
            data_dict=data,
            sampling_rate=sampling_rate,
            out_dir=eeg_out,
            channel='EEG1',
            duration_sec=80,
            start_sec=0,
            file_prefix=f'participant-{pid}_run-{run_id}'
        )
        print(f"Raw EEG excerpt plots saved to {eeg_out}")
    except Exception as e:
        print(f"Could not generate raw EEG excerpt plots: {e}")

    # ------------------------------------------------------------
    # High-transition EEG excerpt (maximize stage transitions)
    # ------------------------------------------------------------
    try:
        pid_ht, run_ht, start_sec_ht, n_trans = _find_high_transition_excerpt(
            meta, sampling_rate=sampling_rate, duration_sec=80, max_runs_scan=max_participants_scan, min_transitions=9
        )
        if pid_ht is not None:
            data_ht = load_participant_data(pid_ht, run_ht)
            eeg_out = out_base / 'raw_eeg'
            _plot_raw_eeg_excerpt(
                data_dict=data_ht,
                sampling_rate=sampling_rate,
                out_dir=eeg_out,
                channel='EEG1',
                duration_sec=80,
                start_sec=start_sec_ht,
                file_prefix=f'high_transition_sub-{pid_ht}_run-{run_ht}_t{int(start_sec_ht)}s',
                subject=pid_ht,
                run=run_ht,
                time=int(start_sec_ht)
            )
            print(f"High-transition excerpt: sub={pid_ht} run={run_ht} start={start_sec_ht:.1f}s transitions={n_trans}")
        else:
            print("Could not find a high-transition excerpt (no labels available in scanned runs).")
    except Exception as e:
        print(f"High-transition excerpt search failed: {e}")

if __name__ == '__main__':
    analyze_dataset_overview()
