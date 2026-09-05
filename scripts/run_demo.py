"""Launch the Streamlit demo."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    app = ROOT / "demo" / "app.py"
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    subprocess.check_call([sys.executable, "-m", "streamlit", "run", str(app), *sys.argv[1:]], env=env)


if __name__ == "__main__":
    main()
