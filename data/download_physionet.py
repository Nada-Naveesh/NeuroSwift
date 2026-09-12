"""Download PhysioNet EEG Motor Movement/Imagery Dataset (EEGMMIDB)."""

import os
from pathlib import Path
import sys
import requests

DATA_RAW_DIR = Path("./data/raw")
BASE_URL = "https://physionet.org/files/eegmmidb/1.0.0/"


def download_file(url: str, dest_path: Path):
    if dest_path.exists() and dest_path.stat().st_size > 1000:
        return
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, stream=True, timeout=25)
    response.raise_for_status()
    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=16384):
            if chunk:
                f.write(chunk)


def download_physionet(num_subjects=109, runs=None):
    """Download PhysioNet motor movement and imagery recordings."""
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("DOWNLOADING PHYSIONET EEG DATASET (EEGMMIDB)")
    print(f"Targeting {num_subjects} subjects, runs: {runs or '3-14 (motor imagery & execution)'}")
    print("=" * 60)

    sample = Path("S001R01.edf")
    if sample.exists() and not (DATA_RAW_DIR / "S001R01.edf").exists():
        (DATA_RAW_DIR / "S001R01.edf").write_bytes(sample.read_bytes())

    if runs is None:
        runs = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

    downloaded = 0
    for subj in range(1, num_subjects + 1):
        subj_str = f"S{subj:03d}"
        for run in runs:
            fname = f"{subj_str}R{run:02d}.edf"
            dest = DATA_RAW_DIR / fname
            if not dest.exists():
                url = f"{BASE_URL}{subj_str}/{fname}"
                try:
                    print(f"Downloading {fname}...")
                    download_file(url, dest)
                    downloaded += 1
                except Exception as e:
                    print(f"Skipping {fname}: {e}")

    count = len(list(DATA_RAW_DIR.glob("*.edf")))
    print(f"\n✅ PhysioNet download complete! Total EDF files in {DATA_RAW_DIR}: {count}")


if __name__ == "__main__":
    download_physionet(num_subjects=5)
