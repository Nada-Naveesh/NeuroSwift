# EEG Motor Movement/Imagery Dataset (EEGMMIDB)

Source: [PhysioNet EEGMMIDB](https://physionet.org/content/eegmmidb/1.0.0/)

Place downloaded EDF files under `data/raw/` (default for `mne.datasets.eegbci`).

```text
python scripts/download_dataset.py --subjects 1-10
python scripts/preprocess.py --imagery-only --no-ica --subjects 1-10
```

Processed arrays are written to `data/processed/trials.npz` with keys `X`, `y`, and optional `subject`.
