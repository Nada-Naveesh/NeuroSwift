"""Map PhysioNet annotations to 5-class motor imagery labels and crop trials."""

from __future__ import annotations

from pathlib import Path

import mne
import numpy as np

from src.config import (
    BOTH_FEET_RUNS,
    CLASS_TO_INDEX,
    LEFT_RIGHT_RUNS,
    N_TIMES,
    SAMPLING_RATE,
    TRIAL_DURATION_SEC,
)


def run_id_from_path(path: str | Path) -> int | None:
    stem = Path(path).stem.lower()
    # PhysioNet files look like S001R04.edf
    if "r" in stem:
        try:
            return int(stem.split("r")[-1])
        except ValueError:
            return None
    return None


def map_annotation_to_class(description: str, run: int | None) -> str | None:
    code = description.strip().upper()
    if code in {"T0", "0"}:
        return "Rest"
    if run is None:
        return None
    if run in LEFT_RIGHT_RUNS:
        if code in {"T1", "1"}:
            return "Left Hand"
        if code in {"T2", "2"}:
            return "Right Hand"
    if run in BOTH_FEET_RUNS:
        if code in {"T1", "1"}:
            return "Both Hands"
        if code in {"T2", "2"}:
            return "Feet"
    return None


def extract_trials(
    raw: mne.io.BaseRaw,
    run: int | None = None,
    tmin: float = 0.0,
    tmax: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return X (n_trials, n_channels, n_times) and y (n_trials,) class indices."""
    tmax = TRIAL_DURATION_SEC if tmax is None else tmax
    events, event_id = mne.events_from_annotations(raw, verbose="ERROR")
    if events.size == 0:
        return np.empty((0, len(raw.ch_names), N_TIMES), dtype=np.float32), np.empty((0,), dtype=np.int64)

    inv = {v: k for k, v in event_id.items()}
    X_list: list[np.ndarray] = []
    y_list: list[int] = []
    sfreq = float(raw.info["sfreq"] or SAMPLING_RATE)
    n_times = int(round((tmax - tmin) * sfreq))
    data = raw.get_data()

    for onset_sample, _, eid in events:
        desc = inv.get(int(eid), str(eid))
        label = map_annotation_to_class(desc, run)
        if label is None:
            continue
        start = int(onset_sample + tmin * sfreq)
        stop = start + n_times
        if start < 0 or stop > data.shape[1]:
            continue
        trial = data[:, start:stop]
        if trial.shape[1] != n_times:
            continue
        if n_times != N_TIMES:
            trial = _resample_trial(trial, N_TIMES)
        X_list.append(trial.astype(np.float32))
        y_list.append(CLASS_TO_INDEX[label])

    if not X_list:
        return np.empty((0, data.shape[0], N_TIMES), dtype=np.float32), np.empty((0,), dtype=np.int64)
    return np.stack(X_list, axis=0), np.asarray(y_list, dtype=np.int64)


def _resample_trial(trial: np.ndarray, n_times: int) -> np.ndarray:
    n_ch, n_old = trial.shape
    old_idx = np.linspace(0, 1, n_old)
    new_idx = np.linspace(0, 1, n_times)
    out = np.empty((n_ch, n_times), dtype=np.float32)
    for ch in range(n_ch):
        out[ch] = np.interp(new_idx, old_idx, trial[ch]).astype(np.float32)
    return out
