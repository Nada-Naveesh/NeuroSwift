"""
NEURALIS: BCI Competition IV 2a Dataset Processor
Adapted for 22 EEG channels, 250 Hz -> 160 Hz resampling, and 4-class motor imagery.
"""

import os
from pathlib import Path
import numpy as np
import scipy.io as sio
from scipy import signal

# Standard 22 EEG channels in BCI Competition IV 2a (10-20 system)
BCI_CHANNELS = [
    'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4',
    'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
    'CP3', 'CP1', 'CPz', 'CP2', 'CP4',
    'P1', 'Pz', 'P2', 'Oz'
]

# 4 motor imagery classes
BCI_CLASSES = ['Left Hand', 'Right Hand', 'Both Feet', 'Tongue']


class BCIIV2aProcessor:
    """
    Process BCI Competition IV 2a dataset.

    Dataset Details:
    - 9 subjects (A01 to A09)
    - 22 EEG channels (monopolar, referenced to left mastoid)
    - 250 Hz original sampling rate
    - 4 classes: Left Hand, Right Hand, Both Feet, Tongue
    - 2 sessions per subject: Training ('T') and Evaluation ('E')
    """

    def __init__(self, sampling_rate=250, target_sampling_rate=160, window_duration=4):
        self.original_sr = sampling_rate
        self.target_sr = target_sampling_rate
        self.window_duration = window_duration
        self.window_samples = int(window_duration * target_sampling_rate)

    def load_mat_file(self, file_path):
        """
        Load BCI IV 2a .mat file safely.
        """
        try:
            data = sio.loadmat(str(file_path))
            return data
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None

    def resample_signal(self, X, original_sr=250, target_sr=160):
        """
        Resample multi-channel EEG from original_sr (250 Hz) to target_sr (160 Hz).
        Input shape: (n_trials, n_channels, n_samples)
        """
        if original_sr == target_sr:
            return X

        n_trials, n_channels, n_samples = X.shape
        new_n_samples = int(n_samples * target_sr / original_sr)

        # Efficient vectorized resample along the last (time) axis
        resampled = signal.resample(X, new_n_samples, axis=-1)
        return resampled

    def bandpass_filter(self, X, lowcut=7, highcut=30, fs=160):
        """
        Apply zero-phase Butterworth bandpass filter (7-30 Hz) for sensorimotor rhythms (mu/beta).
        """
        nyquist = fs / 2.0
        b, a = signal.butter(4, [lowcut / nyquist, highcut / nyquist], btype='band')

        # Efficient zero-phase filtering across trials and channels
        filtered = signal.filtfilt(b, a, X, axis=-1)
        return filtered

    def z_score_normalize(self, X):
        """
        Per-channel z-score normalization along the time dimension for each trial.
        Prevents deep activation vanishing and bias collapse.
        """
        mean = np.mean(X, axis=-1, keepdims=True)
        std = np.std(X, axis=-1, keepdims=True) + 1e-8
        return (X - mean) / std

    def extract_trials(self, data, session_type='T'):
        """
        Extract trials from loaded .mat data.
        Supports both official Graz University struct format and standard 3D arrays.

        BCI IV 2a class mapping:
        1 -> 0: Left Hand
        2 -> 1: Right Hand
        3 -> 2: Both Feet
        4 -> 3: Tongue
        """
        # Check for Graz University BCI IV 2a struct format
        if 'data' in data and hasattr(data['data'], 'shape') and len(data['data']) > 0:
            runs = data['data'][0]
            if len(runs) > 0 and hasattr(runs[0], 'dtype') and runs[0].dtype.names and 'trial' in runs[0].dtype.names:
                trials_list = []
                labels_list = []
                for r in runs:
                    if 'trial' not in r.dtype.names or 'y' not in r.dtype.names:
                        continue
                    trials = r['trial'][0, 0].flatten()
                    labels = r['y'][0, 0].flatten()
                    if len(trials) == 0:
                        continue
                    fs = float(r['fs'][0, 0][0, 0])
                    continuous = r['X'][0, 0][:, :22]
                    for t_start, y_val in zip(trials, labels):
                        s_start = int(t_start + 2.0 * fs)
                        s_end = s_start + int(self.window_duration * fs)
                        if s_end <= len(continuous):
                            epoch = continuous[s_start:s_end, :].T  # (22, samples)
                            trials_list.append(epoch)
                            labels_list.append(int(y_val) - 1 if int(y_val) >= 1 else int(y_val))
                if len(trials_list) > 0:
                    return np.array(trials_list, dtype=np.float32), np.array(labels_list, dtype=np.int64)

        # Look for standard keys 'X' and 'y'
        if 'X' in data:
            X = data['X']
        elif 'data' in data and not isinstance(data['data'][0], np.void):
            X = data['data']
        else:
            valid_keys = [k for k in data.keys() if not k.startswith('__')]
            if valid_keys:
                X = data[valid_keys[0]]
            else:
                raise ValueError("No trial data array found in .mat dictionary.")

        if 'y' in data:
            y = data['y'].flatten()
        elif 'labels' in data:
            y = data['labels'].flatten()
        elif 'label' in data:
            y = data['label'].flatten()
        else:
            raise ValueError("No label vector found in .mat dictionary.")

        # Ensure X is 3D: (n_trials, 22, n_samples)
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 2:
            X = np.expand_dims(X, axis=0)

        # Convert 1-indexed (1, 2, 3, 4) to 0-indexed (0, 1, 2, 3)
        y = np.asarray(y, dtype=np.int64)
        if y.min() >= 1:
            y = y - 1

        return X, y

    def process_subject(self, file_path, session_type='T'):
        """
        Complete processing pipeline for one subject file:
        Load -> Extract -> Resample -> Bandpass Filter -> Z-Score Normalization
        Handles continuous filtering on Graz format to eliminate boundary filter transients.
        """
        data = self.load_mat_file(file_path)
        if data is None:
            return None, None

        # Check for Graz University BCI IV 2a struct format
        if 'data' in data and hasattr(data['data'], 'shape') and len(data['data']) > 0:
            runs = data['data'][0]
            if len(runs) > 0 and hasattr(runs[0], 'dtype') and runs[0].dtype.names and 'trial' in runs[0].dtype.names:
                trials_list = []
                labels_list = []
                for r in runs:
                    if 'trial' not in r.dtype.names or 'y' not in r.dtype.names:
                        continue
                    trials = r['trial'][0, 0].flatten()
                    labels = r['y'][0, 0].flatten()
                    if len(trials) == 0:
                        continue
                    fs = float(r['fs'][0, 0][0, 0])
                    continuous = r['X'][0, 0][:, :22]
                    
                    # Zero-phase Butterworth bandpass filter (7-30 Hz) on continuous run
                    nyq = fs / 2.0
                    b, a = signal.butter(4, [7.0 / nyq, 30.0 / nyq], btype='band')
                    filtered = signal.filtfilt(b, a, continuous, axis=0)

                    for t_start, y_val in zip(trials, labels):
                        # Visual cue starts 2.0s after fixation start
                        s_start = int(t_start + 2.0 * fs)
                        s_end = s_start + int(self.window_duration * fs)
                        if s_end <= len(filtered):
                            epoch = filtered[s_start:s_end, :]  # (1000, 22)
                            if fs != self.target_sr:
                                epoch = signal.resample(epoch, self.window_samples, axis=0)
                            epoch = epoch.T  # (22, 640)
                            mean = np.mean(epoch, axis=-1, keepdims=True)
                            std = np.std(epoch, axis=-1, keepdims=True) + 1e-8
                            normed = (epoch - mean) / std
                            trials_list.append(normed)
                            labels_list.append(int(y_val) - 1 if int(y_val) >= 1 else int(y_val))

                if len(trials_list) > 0:
                    X = np.array(trials_list, dtype=np.float32)
                    y = np.array(labels_list, dtype=np.int64)
                    return X, y

        X, y = self.extract_trials(data, session_type)

        # Resample from 250 Hz to target_sr (160 Hz)
        X = self.resample_signal(X, self.original_sr, self.target_sr)

        # Bandpass filter (7-30 Hz)
        X = self.bandpass_filter(X, lowcut=7, highcut=30, fs=self.target_sr)

        # Per-channel Z-score normalization
        X = self.z_score_normalize(X)

        return X, y


def generate_sample_bci_data(output_dir, n_subjects=9, trials_per_class=30, random_seed=42):
    """
    Generate realistic benchmark sample BCI IV 2a data (22 channels, 160 Hz, 4 classes)
    emulating mu (10 Hz) and beta (20 Hz) sensorimotor desynchronization/synchronization.
    Allows immediate end-to-end testing of the pipeline without external downloads.
    """
    np.random.seed(random_seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_classes = 4  # Left Hand, Right Hand, Both Feet, Tongue
    n_channels = 22
    n_samples = 640  # 4.0 seconds at 160 Hz
    fs = 160.0
    t = np.linspace(0, 4.0, n_samples)

    all_X = []
    all_y = []
    all_subjects = []

    # Motor cortex channel indices in BCI IV 2a montage:
    # C3 = 7, Cz = 9, C4 = 11
    ch_c3, ch_cz, ch_c4 = 7, 9, 11

    for subject in range(1, n_subjects + 1):
        subj_X = []
        subj_y = []
        for cls in range(n_classes):
            for _ in range(trials_per_class):
                # Baseline 1/f-like background neural noise
                noise = np.random.randn(n_channels, n_samples) * 0.4
                signal_trial = noise.copy()

                # Add baseline idling mu (10 Hz) and beta (20 Hz)
                mu = np.sin(2 * np.pi * 10.0 * t)
                beta = np.sin(2 * np.pi * 20.0 * t)

                # Class-specific contralateral ERD / ERS signatures:
                if cls == 0:  # Left Hand -> Contralateral right hemisphere C4 ERD, ipsilateral C3 ERS
                    signal_trial[ch_c4] += 0.2 * mu
                    signal_trial[ch_c3] += 0.9 * mu + 0.5 * beta
                elif cls == 1:  # Right Hand -> Contralateral left hemisphere C3 ERD, ipsilateral C4 ERS
                    signal_trial[ch_c3] += 0.2 * mu
                    signal_trial[ch_c4] += 0.9 * mu + 0.5 * beta
                elif cls == 2:  # Both Feet -> Vertex Cz ERD/modulation
                    signal_trial[ch_cz] += 1.1 * beta + 0.3 * mu
                elif cls == 3:  # Tongue -> Bilateral temporal/frontal modulation
                    signal_trial[ch_c3] += 0.6 * beta
                    signal_trial[ch_c4] += 0.6 * beta

                # Z-score normalize
                mean = np.mean(signal_trial, axis=-1, keepdims=True)
                std = np.std(signal_trial, axis=-1, keepdims=True) + 1e-8
                signal_trial = (signal_trial - mean) / std

                subj_X.append(signal_trial)
                subj_y.append(cls)

        subj_X = np.array(subj_X, dtype=np.float32)
        subj_y = np.array(subj_y, dtype=np.int64)

        all_X.append(subj_X)
        all_y.append(subj_y)
        all_subjects.extend([subject] * len(subj_y))

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    subjects = np.array(all_subjects)

    np.save(output_dir / "X_bci.npy", X)
    np.save(output_dir / "y_bci.npy", y)
    np.save(output_dir / "subjects_bci.npy", subjects)

    print(f"Generated {len(X)} benchmark BCI IV 2a trials across {n_subjects} subjects.")
    print(f"Saved to {output_dir}")
    return X, y, subjects


def process_all_bci_subjects(data_dir, output_dir):
    """
    Process all 9 subjects from BCI IV 2a dataset from raw .mat files.
    """
    processor = BCIIV2aProcessor()

    all_X = []
    all_y = []
    all_subjects = []

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("NEURALIS: PROCESSING BCI COMPETITION IV 2a DATASET")
    print("=" * 70)

    for subject in range(1, 10):  # 9 subjects
        for session_type in ['T', 'E']:
            file_path = data_dir / f"A{subject:02d}{session_type}.mat"

            if not file_path.exists():
                continue

            print(f"Processing Subject {subject:02d} Session {session_type}...")
            X, y = processor.process_subject(file_path, session_type)

            if X is not None and len(X) > 0:
                all_X.append(X)
                all_y.append(y)
                all_subjects.extend([subject] * len(y))
                print(f"  Extracted {len(y)} trials")

    if len(all_X) == 0:
        print("No raw .mat files found in data directory.")
        return None, None, None

    # Concatenate all trials
    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    subjects = np.array(all_subjects)

    print("\n" + "=" * 70)
    print("BCI IV 2a PROCESSING COMPLETE")
    print("=" * 70)
    print(f"Total trials: {len(X)}")
    print(f"EEG channels: {X.shape[1]}")
    print(f"Time samples: {X.shape[2]}")
    print(f"Classes: {np.unique(y)}")
    print(f"Class distribution: {np.bincount(y)}")
    print(f"Subjects: {np.unique(subjects)}")
    print("=" * 70)

    # Save to disk
    np.save(output_dir / "X_bci.npy", X)
    np.save(output_dir / "y_bci.npy", y)
    np.save(output_dir / "subjects_bci.npy", subjects)

    print(f"Data saved to {output_dir}")
    return X, y, subjects


if __name__ == "__main__":
    data_dir = "./data/bci_iv_2a/"
    output_dir = "./data/processed_bci/"

    X, y, subjects = process_all_bci_subjects(data_dir, output_dir)
