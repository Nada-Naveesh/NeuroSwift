"""
Run complete NeuroSwift pipeline
"""

import os
import subprocess
import sys

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
os.environ["PYTHONIOENCODING"] = "utf-8"


def run_command(cmd):
    print(f"\n> {cmd}")
    # Replace generic 'python' or 'pip' with the current running interpreter to ensure .venv consistency
    if cmd.startswith("python "):
        cmd = f'"{sys.executable}" ' + cmd[7:]
    elif cmd.startswith("pip "):
        cmd = f'"{sys.executable}" -m pip ' + cmd[4:]
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    return result.returncode == 0


def main():
    print("=" * 60)
    print("NEUROSWIFT: Complete Pipeline")
    print("=" * 60)

    # Step 1: Install dependencies
    print("\n[1/4] Installing dependencies...")
    run_command("pip install -r requirements.txt")

    # Step 2: Download dataset
    print("\n[2/4] Downloading PhysioNet dataset...")
    run_command("python scripts/download_physionet.py")

    # Step 3: Preprocess data
    print("\n[3/4] Preprocessing EEG data...")
    run_command("python preprocessing/signal_processor.py")

    # Step 4: Train model
    print("\n[4/4] Training NeuroSwift model...")
    run_command("python training/train.py")

    print("\n" + "=" * 60)
    print("[OK] NEUROSWIFT PIPELINE COMPLETE!")
    print("=" * 60)
    print("\nTo run the demo:")
    print("streamlit run demo/app.py")


if __name__ == "__main__":
    main()
