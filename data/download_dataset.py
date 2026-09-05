"""Download PhysioNet EEG Motor Movement/Imagery Dataset (EEGMMIDB)."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import urllib.request
import requests
from tqdm import tqdm

from src.config import DATA_RAW_DIR, EXCLUDED_SUBJECTS, IMAGERY_RUNS, PHYSIONET_N_SUBJECTS


def download_file_fallback(url: str, filepath: str) -> None:
    """Download a file with requests/urllib fallback if wget is unavailable."""
    try:
        import wget
        wget.download(url, out=filepath)
    except Exception:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)


def download_physionet(
    subjects: list[int] | range | None = None,
    runs: list[int] | tuple[int, ...] | None = None,
    output_dir: str = "./data/raw/",
) -> None:
    """Download PhysioNet EEG Motor Imagery Dataset (S001 to S109, R01 to R14)."""
    base_url = "https://physionet.org/files/eegmmidb/1.0.0/"
    os.makedirs(output_dir, exist_ok=True)
    subjects = list(subjects or range(1, 110))
    runs = list(runs or range(1, 15))

    print("Downloading PhysioNet EEG Motor Imagery Dataset...")

    for subject in subjects:
        if subject in EXCLUDED_SUBJECTS:
            continue
        subject_id = f"S{subject:03d}"
        for run in runs:
            run_id = f"R{run:02d}"
            filename = f"{subject_id}{run_id}.edf"
            filepath = os.path.join(output_dir, filename)
            if not os.path.exists(filepath):
                file_url = f"{base_url}{subject_id}/{filename}"
                try:
                    print(f"Downloading {filename}...")
                    download_file_fallback(file_url, filepath)
                except Exception as e:
                    print(f"Error downloading {filename}: {e}")

    print("\nDownload complete!")


def parse_subject_list(spec: str) -> list[int]:
    subjects: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            subjects.extend(range(int(a), int(b) + 1))
        else:
            subjects.append(int(part))
    return sorted(set(subjects))


def main() -> None:
    parser = argparse.ArgumentParser(description="Download PhysioNet EEGMMIDB.")
    parser.add_argument("--subjects", default="1-10", help="e.g. 1-109 or 1,2,5")
    parser.add_argument("--imagery-only", action="store_true", help="Download imagined-movement runs only")
    parser.add_argument("--data-dir", type=str, default=str(DATA_RAW_DIR))
    args = parser.parse_args()

    subjs = parse_subject_list(args.subjects)
    runs = IMAGERY_RUNS if args.imagery_only else tuple(range(1, 15))
    download_physionet(subjects=subjs, runs=runs, output_dir=args.data_dir)


if __name__ == "__main__":
    download_physionet()
