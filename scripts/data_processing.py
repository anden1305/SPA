
from pathlib import Path
import numpy as np
import pandas as pd
import ast
import re
import pyedflib


DATA_ROOT = Path("data") / "ds006366"
OUTPUT_DIR = Path("results") / "data_exploration"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NEW_DATA_DIR = Path("data") / "ds006366_processed"
NEW_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Use explicit mapping from task-sleep_events.json
STAGE_MAP = {
    1: "Wake",
    2: "NREM",
    3: "REM",
    4: "Artifact"
}

LAB_EEG_PLACEMENTS = {
    'lab_1': ['EEG IFPD'],
    'lab_2': ['EEG P', 'EEG P', 'EEG F', 'EEG F'],
    'lab_3': ['EEG P', 'EEG F'],
    'lab_4': ['EEG PFCF'],
    'lab_5': ['EEG P', 'EEG F']
}

def get_participant_data():
    participants_path = DATA_ROOT / "participants.tsv"
    participants_df = pd.read_csv(participants_path, sep="\t")
    return participants_df

def make_lab_data():
    labs_path = DATA_ROOT / "labs.tsv"
    labs_df = pd.read_csv(labs_path, sep="\t")
    labs_df['channels'] = labs_df['channels'].apply(ast.literal_eval)
    unique_channels = sorted(set(ch for sublist in labs_df['channels'] for ch in sublist))
    def channel_positions(channel_list):
        return {ch: (int(channel_list.index(ch)) if ch in channel_list else None) for ch in unique_channels}
    df_positions = labs_df['channels'].apply(channel_positions).apply(pd.Series)
    lab_data = pd.concat([labs_df, df_positions], axis=1).drop(columns=['channels'])
    return lab_data
    
EEG_RE    = re.compile(r"_run-(\d+)_eeg\.edf$", re.IGNORECASE)
EVENTS_RE = re.compile(r"_run-(\d+)_events\.tsv$", re.IGNORECASE)

def make_run_data_for_participant(subject: str):
    subject_path = DATA_ROOT / subject / 'eeg'
    
    runs = {}
    num_runs = 0

    for p in subject_path.iterdir():
        if not p.is_file():
            continue
        name = p.name

        m = EEG_RE.search(name)
        if m:
            run = int(m.group(1))
            runs[f'path_eeg_{run}'] = str(p)
        
        m = EVENTS_RE.search(name)
        if m:
            num_runs += 1
            run = int(m.group(1))
            runs[f'path_events_{run}'] = str(p)
    
    runs['num_runs'] = num_runs
    runs['participant_id'] = subject
    
    return runs

def make_run_data(subjects: list[str]):
    data = []
    for subject in subjects:
        data.append(make_run_data_for_participant(subject))
    return pd.DataFrame(data)

def make_participant_data():
    participants = get_participant_data()
    lab_data = make_lab_data()
    df_merged = pd.merge(participants, lab_data, on='lab', how='left')
    run_data = make_run_data(df_merged['participant_id'].tolist())
    df_merged_2 = pd.merge(df_merged, run_data, on='participant_id', how='left')
    df_merged_2['sampling_frequency'] = 128
    return df_merged_2, lab_data


def extract_signals_from_edf(edf_path, return_time=True):
    """
    Returns a dict keyed by channel label.
    Each value is {'data': np.ndarray, 'fs': float, 't': np.ndarray or None, 'unit': str}.
    - data: physical values (scaled), length = number of samples for that channel
    - fs: sampling rate (Hz) for that channel
    - t: time vector in seconds (optional)
    - unit: physical dimension from header (e.g., 'uV')
    """
    out = {}

    with pyedflib.EdfReader(edf_path) as f:
        n_signals = f.signals_in_file
        labels = [lbl.strip() if lbl else f"ch{idx}" for idx, lbl in enumerate(f.getSignalLabels())]
        pds = [f.getPhysicalDimension(i) or "" for i in range(n_signals)]
        srs = [f.getSampleFrequency(i) for i in range(n_signals)]

        # Ensure unique keys even if labels repeat
        seen = {}
        def unique(label):
            k = label
            if k in seen:
                seen[k] += 1
                k = f"{label}-{seen[label]}"
            else:
                seen[k] = 0
            return k

        for i in range(n_signals):
            key = unique(labels[i])
            data = f.readSignal(i).astype(np.float64)  # physical units
            fs = float(srs[i])
            t = (np.arange(data.size) / fs) if return_time else None
            out[key] = {"data": data, "fs": fs, "t": t, "unit": pds[i]}

    return out


def load_stage_labels(path: str, n_samples: int, fs: int) -> np.ndarray:
    """
    Read events.tsv and build a sample-wise stage vector at `fs` Hz.
    For each row, extend the stage for fs*duration samples (row order; onset is ignored).
    If the result is shorter than `n_samples`, repeat the last stage to reach it.
    If longer, truncate to exactly `n_samples`.

    Returns: 1-D np.ndarray (int16) of length n_samples.
    """
    df = pd.read_csv(path, sep="\t")
    # keep it simple: just use these two columns; drop rows with missing values
    df = df[["duration", "stage"]].dropna()

    # how many samples per row
    reps = np.rint(df["duration"].to_numpy(float) * fs).astype(int)
    stages = df["stage"].to_numpy(int).astype(np.int16)

    labels = np.repeat(stages, reps)

    if labels.size == 0:
        # no rows or all durations ~0: fill with -1
        return np.full(n_samples, -1, dtype=np.int16)

    if labels.size < n_samples:
        tail = np.full(n_samples - labels.size, labels[-1], dtype=np.int16)
        labels = np.concatenate([labels, tail])
    elif labels.size > n_samples:
        labels = labels[:n_samples]

    return labels

def extract_data(participant_data: pd.DataFrame):
    metadata = []
    for participant in participant_data.iloc:
        participant_id: str = participant['participant_id']
        lab = participant['lab']
        num_runs = participant['num_runs']
        eeg1 = pd.notna(participant['EEG1'])
        eeg2 = pd.notna(participant['EEG2'])
        eeg3 = pd.notna(participant['EEG3'])
        eeg4 = pd.notna(participant['EEG4'])
        emg = pd.notna(participant['EMG'])
        eeg_type_1 = LAB_EEG_PLACEMENTS[lab][0] if eeg1 else None
        eeg_type_2 = LAB_EEG_PLACEMENTS[lab][1] if eeg2 else None
        eeg_type_3 = LAB_EEG_PLACEMENTS[lab][2] if eeg3 else None
        eeg_type_4 = LAB_EEG_PLACEMENTS[lab][3] if eeg4 else None
        for i in range(num_runs):
            run = i+1
            new_path = NEW_DATA_DIR / participant_id / str(run)
            new_path.mkdir(parents=True, exist_ok=True)
            edf_path = participant[f'path_eeg_{run}']
            signals = extract_signals_from_edf(edf_path)
            samples = None
            for signal in signals:
                signal_path = new_path / (signal + ".npy")
                signal_data = signals[signal]['data']
                if not samples:
                    samples = len(signal_data)
                np.save(signal_path, np.asarray(signal_data, dtype=np.float32))
            events_path = participant[f'path_events_{run}']
            labels = load_stage_labels(events_path, n_samples=samples, fs=128)
            new_events_path = new_path / "labels.npy"
            np.save(new_events_path, np.asarray(labels, dtype=np.int16))
            metadata.append(
                {
                    'participant_id': participant_id,
                    'lab': lab,
                    'run': run,
                    'fs': 128,
                    'samples': samples,
                    'seconds': samples/128,
                    'EEG1': eeg1,
                    'EEG2': eeg2,
                    'EEG3': eeg3,
                    'EEG4': eeg4,
                    'EMG': emg,
                    'EEG1_TYPE': eeg_type_1,
                    'EEG2_TYPE': eeg_type_2,
                    'EEG3_TYPE': eeg_type_3,
                    'EEG4_TYPE': eeg_type_4,
                    'path': str(new_path),
                }
            )
    return pd.DataFrame(metadata)

def save_csv(dataframe: pd.DataFrame, filename: str):
    dataframe.to_csv(NEW_DATA_DIR / filename)

if __name__ == "__main__":
    
    participant_data, lab_data = make_participant_data()
    metadata = extract_data(participant_data=participant_data)
    
    save_csv(participant_data, 'participant_data.csv')
    save_csv(lab_data, 'lab_data.csv')
    save_csv(metadata, 'metadata.csv')