import os
from pathlib import Path
import warnings
import numpy as np
import mne
from mne.io import read_raw_edf

warnings.filterwarnings('ignore')


class EEGSignalProcessor:
    def __init__(self, sampling_rate=160, window_duration=4):
        self.sampling_rate = sampling_rate
        self.window_duration = window_duration
        self.window_samples = sampling_rate * window_duration
        
    def load_edf(self, file_path):
        try:
            raw = read_raw_edf(file_path, preload=True, verbose='ERROR')
            return raw
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    
    def preprocess_raw(self, raw):
        try:
            mapping = {name: name.replace(".", "").upper() for name in raw.ch_names}
            raw.rename_channels(mapping, verbose='ERROR')
            montage = mne.channels.make_standard_montage('standard_1020')
            raw.set_montage(montage, on_missing='ignore')
        except Exception:
            pass
        raw.filter(7.0, 30.0, fir_design='firwin', verbose='ERROR')
        if raw.info['sfreq'] != self.sampling_rate:
            raw.resample(self.sampling_rate, verbose='ERROR')
        return raw

    def process_file(self, file_path):
        raw = self.load_edf(file_path)
        if raw is None:
            return None, None
        raw = self.preprocess_raw(raw)
        stem = Path(file_path).stem.upper()
        run = None
        if "R" in stem:
            try:
                run = int(stem.split("R")[-1])
            except ValueError:
                run = None
        return self.extract_trials(raw, run=run)
    
    def extract_trials(self, raw, run=None):
        try:
            events, event_id = mne.events_from_annotations(raw, verbose='ERROR')
        except Exception:
            events, event_id = np.empty((0, 3), dtype=int), {}
            
        both_feet_runs = {5, 6, 9, 10, 13, 14}
        if run in both_feet_runs:
            label_map = {'T0': 4, 'T1': 2, 'T2': 3, '0': 4, '1': 2, '2': 3}
        else:
            label_map = {'T0': 4, 'T1': 0, 'T2': 1, '0': 4, '1': 0, '2': 1}
        inv_event_id = {v: k for k, v in event_id.items()} if event_id else {}
        
        if len(events) == 0:
            data = raw.get_data()
            n_samples = data.shape[1]
            if n_samples >= self.window_samples:
                X = np.stack(
                    [data[:, i : i + self.window_samples] for i in range(0, n_samples - self.window_samples + 1, self.window_samples)],
                    axis=0,
                )
                y = np.full((len(X),), 4, dtype=np.int64)
            else:
                return np.empty((0, data.shape[0], self.window_samples)), np.empty((0,), dtype=np.int64)
        else:
            epochs = mne.Epochs(
                raw, events, event_id,
                tmin=0, tmax=self.window_duration,
                baseline=None, preload=True, verbose='ERROR'
            )
            
            X = epochs.get_data()
            if X.shape[-1] > self.window_samples:
                X = X[:, :, :self.window_samples]
            y_raw = epochs.events[:, -1]
            
            y_mapped = []
            for code in y_raw:
                desc = inv_event_id.get(int(code), str(code))
                y_mapped.append(label_map.get(desc, -1))
            
            y_mapped = np.array(y_mapped, dtype=np.int64)
            valid_idx = y_mapped >= 0
            X = X[valid_idx]
            y = y_mapped[valid_idx]
        
        # Standardize channels to 64 channels
        if X.ndim == 3 and X.shape[1] != 64:
            X_pad = np.zeros((X.shape[0], 64, X.shape[2]), dtype=np.float32)
            n_ch = min(X.shape[1], 64)
            X_pad[:, :n_ch, :] = X[:, :n_ch, :]
            X = X_pad
            
        # Z-score normalization per channel
        for i in range(X.shape[0]):
            for ch in range(X.shape[1]):
                std = X[i, ch].std()
                if std > 1e-8:
                    X[i, ch] = (X[i, ch] - X[i, ch].mean()) / (std + 1e-8)
                else:
                    X[i, ch] = X[i, ch] - X[i, ch].mean()
        
        return X.astype(np.float32), y.astype(np.int64)


def pad_or_trim_channels(data: np.ndarray, n_channels: int = 64) -> np.ndarray:
    """Ensure array is (n_channels, n_times)."""
    if data.ndim != 2:
        raise ValueError(f"Expected 2-D EEG array, got shape {data.shape}")
    if data.shape[0] > data.shape[1] and data.shape[1] <= n_channels:
        data = data.T
    n_ch, n_t = data.shape
    if n_ch == n_channels:
        return data.astype(np.float32, copy=False)
    out = np.zeros((n_channels, n_t), dtype=np.float32)
    n = min(n_ch, n_channels)
    out[:n] = data[:n]
    return out


def load_raw_edf(path, preload: bool = True):
    return mne.io.read_raw_edf(str(path), preload=preload, verbose='ERROR')


def preprocess_raw(raw, use_ica: bool = False):
    processor = EEGSignalProcessor()
    return processor.preprocess_raw(raw)


def process_all_files(data_dir='./data/raw', num_subjects=5):
    processor = EEGSignalProcessor()
    all_X = []
    all_y = []
    
    for subject in range(1, num_subjects + 1):
        subject_id = f"S{subject:03d}"
        for run in range(3, 15):
            run_id = f"R{run:02d}"
            filename = f"{subject_id}{run_id}.edf"
            filepath = os.path.join(data_dir, filename)
            
            if os.path.exists(filepath):
                print(f"Processing {filename}...")
                X, y = processor.process_file(filepath)
                if X is not None and len(X) > 0:
                    all_X.append(X)
                    all_y.append(y)
    
    if len(all_X) == 0:
        return None, None
    
    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    
    return X, y


if __name__ == "__main__":
    out_dir = Path("./data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    X, y = process_all_files("./data/raw", num_subjects=5)
    
    # If no raw files processed yet, check for existing trials.npz or generate fallback
    if X is None or len(X) == 0:
        npz_file = out_dir / "trials.npz"
        if npz_file.exists():
            payload = np.load(npz_file)
            X, y = payload["X"], payload["y"]
            print(f"Loaded existing {len(X)} trials from {npz_file}")
        else:
            from data.synthetic import generate_dataset
            from preprocessing.feature_engineer import zscore_batch
            print("No raw EDF files found. Generating balanced demo dataset...")
            X, y = generate_dataset(n_per_class=100)
            X = zscore_batch(X)
            np.savez_compressed(npz_file, X=X, y=y)
            
    np.save(out_dir / "X.npy", X)
    np.save(out_dir / "y.npy", y)
    print(f"Saved {len(X)} processed trials to ./data/processed/X.npy and y.npy")
