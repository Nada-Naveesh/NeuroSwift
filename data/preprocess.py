"""Offline preprocessing: EDF → filtered, segmented, z-scored trials."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.download_dataset import parse_subject_list
from preprocessing.feature_engineer import ensure_trial_shape, zscore_trial
from preprocessing.signal_processor import load_raw_edf, pad_or_trim_channels, preprocess_raw
from preprocessing.trial_extractor import extract_trials, run_id_from_path
from src.config import DATA_PROCESSED_DIR, DATA_RAW_DIR, EXCLUDED_SUBJECTS, IMAGERY_RUNS


def find_edf_files(data_dir: Path, subjects: list[int] | None, imagery_only: bool) -> list[Path]:
    files = sorted(data_dir.rglob("*.edf"))
    out: list[Path] = []
    for path in files:
        run = run_id_from_path(path)
        if imagery_only and run not in IMAGERY_RUNS:
            continue
        if subjects is not None:
            stem = path.stem.upper()
            try:
                subj = int(stem.split("R")[0].replace("S", ""))
            except ValueError:
                continue
            if subj in EXCLUDED_SUBJECTS or subj not in subjects:
                continue
        out.append(path)
    return out


def process_file(path: Path, use_ica: bool) -> tuple[np.ndarray, np.ndarray]:
    raw = load_raw_edf(path)
    raw = preprocess_raw(raw, use_ica=use_ica)
    run = run_id_from_path(path)
    X, y = extract_trials(raw, run=run)
    if len(X) == 0:
        return X, y
    X = np.stack([zscore_trial(ensure_trial_shape(pad_or_trim_channels(t))) for t in X], axis=0)
    return X, y


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess PhysioNet EDF files.")
    parser.add_argument("--data-dir", type=Path, default=DATA_RAW_DIR)
    parser.add_argument("--out", type=Path, default=DATA_PROCESSED_DIR / "trials.npz")
    parser.add_argument("--subjects", default=None, help="e.g. 1-10 (optional filter)")
    parser.add_argument("--imagery-only", action="store_true")
    parser.add_argument("--no-ica", action="store_true", help="Skip ICA (much faster)")
    args = parser.parse_args()

    subjects = parse_subject_list(args.subjects) if args.subjects else None
    files = find_edf_files(args.data_dir, subjects, args.imagery_only)
    if not files:
        raise SystemExit(f"No EDF files found under {args.data_dir}. Run scripts/download_dataset.py first.")

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for path in tqdm(files, desc="Preprocessing"):
        try:
            X, y = process_file(path, use_ica=not args.no_ica)
        except Exception as exc:
            print(f"Skip {path.name}: {exc}")
            continue
        if len(y) == 0:
            continue
        xs.append(X)
        ys.append(y)

    if not xs:
        raise SystemExit("No trials extracted. Check annotations and run IDs.")

    X = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, X=X, y=y)
    counts = {int(c): int((y == c).sum()) for c in np.unique(y)}
    print(f"Saved {X.shape[0]} trials {X.shape[1:]} -> {args.out}")
    print(f"Class counts: {counts}")


if __name__ == "__main__":
    main()
