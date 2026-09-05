"""CLI wrappers live here; implementation is in package modules."""

from __future__ import annotations

import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(module_file: str) -> None:
    runpy.run_path(str(ROOT / module_file), run_name="__main__")


if __name__ == "__main__":
    raise SystemExit("Use a specific script: download_dataset.py, preprocess.py, train.py, evaluate.py")
