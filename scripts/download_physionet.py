"""Download PhysioNet EEG Motor Movement/Imagery Dataset (EEGMMIDB)."""

import os
from pathlib import Path
import sys
import requests

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

DATA_RAW_DIR = Path("./data/raw")
BASE_URL = "https://physionet.org/files/eegmmidb/1.0.0/"


def download_file(url: str, dest_path: Path):
    if dest_path.exists() and dest_path.stat().st_size > 1000:
        return
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, stream=True, timeout=20)
    response.raise_for_status()
    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=16384):
            if chunk:
                f.write(chunk)


def download_physionet(num_subjects=1, runs=None):
    print("=" * 60)
    print("DOWNLOADING PHYSIONET EEG DATASET (EEGMMIDB)")
    print("=" * 60)
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)

    sample = Path("S001R01.edf")
    if sample.exists() and not (DATA_RAW_DIR / "S001R01.edf").exists():
        (DATA_RAW_DIR / "S001R01.edf").write_bytes(sample.read_bytes())
        print("Copied bundled sample S001R01.edf into data/raw/")

    # Imagery runs (4, 6, 8, 10, 12, 14) and execution runs (3, 5, 7, 9, 11, 13)
    if runs is None:
        runs = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

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
                except Exception as e:
                    print(f"Skipping {fname}: {e}")

    count = len(list(DATA_RAW_DIR.glob("*.edf")))
    print(f"\n[OK] PhysioNet download complete! Total files in {DATA_RAW_DIR}: {count}")


if __name__ == "__main__":
    download_physionet(num_subjects=1)
