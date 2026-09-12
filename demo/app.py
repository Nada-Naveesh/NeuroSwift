import os
from pathlib import Path
import sys
import tempfile
import matplotlib.pyplot as plt
import mne
import numpy as np
import streamlit as st
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

st.set_page_config(page_title="NeuroSwift", page_icon="🧠", layout="wide")

st.title("🧠 NeuroSwift")
st.subheader("Motor Imagery Classification from EEG Signals")


def normalize_trial(trial: np.ndarray) -> np.ndarray:
    """Ensure trial is z-score normalized per channel."""
    t = trial.astype(np.float32, copy=True)
    if t.ndim == 2:
        for ch in range(t.shape[0]):
            std = t[ch].std()
            if std > 1e-8:
                t[ch] = (t[ch] - t[ch].mean()) / (std + 1e-8)
            else:
                t[ch] = t[ch] - t[ch].mean()
    elif t.ndim == 3:
        for i in range(t.shape[0]):
            for ch in range(t.shape[1]):
                std = t[i, ch].std()
                if std > 1e-8:
                    t[i, ch] = (t[i, ch] - t[i, ch].mean()) / (std + 1e-8)
                else:
                    t[i, ch] = t[i, ch] - t[i, ch].mean()
    return t


def process_raw_to_trials(raw, filename: str):
    """Filter raw EEG, extract motor imagery epochs, and normalize."""
    raw.filter(7.0, 30.0, fir_design="firwin", verbose="ERROR")
    events, event_id = mne.events_from_annotations(raw, verbose="ERROR")

    stem = Path(filename).stem.upper()
    run = None
    if "R" in stem:
        try:
            run = int(stem.split("R")[-1])
        except ValueError:
            run = None

    both_feet_runs = {5, 6, 9, 10, 13, 14}
    if run in both_feet_runs:
        label_map = {"T0": 4, "T1": 2, "T2": 3, "0": 4, "1": 2, "2": 3}
    else:
        label_map = {"T0": 4, "T1": 0, "T2": 1, "0": 4, "1": 0, "2": 1}

    inv_event_id = {v: k for k, v in event_id.items()} if event_id else {}

    if len(events) == 0:
        data = raw.get_data()
        n_samples = data.shape[1]
        if n_samples >= 640:
            X = np.stack(
                [data[:, i : i + 640] for i in range(0, n_samples - 640 + 1, 640)],
                axis=0,
            )
            y = np.full((len(X),), 4, dtype=np.int64)
        else:
            X, y = np.empty((0, 64, 640)), np.empty((0,))
    else:
        epochs = mne.Epochs(
            raw, events, event_id, tmin=0, tmax=4, baseline=None, preload=True, verbose="ERROR"
        )
        X = epochs.get_data()
        if X.shape[-1] > 640:
            X = X[:, :, :640]
        y_raw = epochs.events[:, -1]
        y_mapped = np.array([label_map.get(inv_event_id.get(int(c), str(c)), -1) for c in y_raw])
        valid_idx = y_mapped >= 0
        X = X[valid_idx]
        y = y_mapped[valid_idx]

    # Standardize to 64 channels
    if X.ndim == 3 and X.shape[1] != 64:
        X_pad = np.zeros((X.shape[0], 64, X.shape[2]), dtype=np.float32)
        n_ch = min(X.shape[1], 64)
        X_pad[:, :n_ch, :] = X[:, :n_ch, :]
        X = X_pad

    # Per-channel z-score normalization
    X = normalize_trial(X)
    return X.astype(np.float32), y.astype(np.int64)


# Load model function
@st.cache_resource
def load_models():
    from models.neuroswift import NeuroSwiftModel
    from models.ensemble import EnsembleModel
    from training.config import CONFIG

    candidates = [
        "best_model_improved.pt",
        "best_model_final.pt",
        "checkpoints/best_model_improved.pt",
        "checkpoints/best_model_effatt.pt",
        "checkpoints/best_model_final.pt",
    ]
    single_model = None
    weights_path = None
    for c in candidates:
        if os.path.exists(c):
            weights_path = c
            break

    try:
        single_model = NeuroSwiftModel(CONFIG)
        if weights_path is not None:
            state = torch.load(weights_path, map_location="cpu")
            if isinstance(state, dict) and "model_state_dict" in state:
                single_model.load_state_dict(state["model_state_dict"], strict=False)
            else:
                single_model.load_state_dict(state, strict=False)
            print(f"Loaded single model weights from {weights_path}")
        single_model.eval()
    except Exception as e:
        print(f"Error loading single model: {e}")

    # Load ensemble
    ensemble = None
    try:
        ensemble = EnsembleModel(CONFIG, num_models=5)
        loaded_count = ensemble.load_weights("./checkpoints")
        print(f"Loaded {loaded_count} ensemble checkpoints")
    except Exception as e:
        print(f"Error initializing ensemble: {e}")

    return single_model, ensemble, weights_path


single_model, ensemble_model, loaded_path = load_models()

# Sidebar
with st.sidebar:
    st.header("⚙️ Model Architecture")
    model_choice = st.selectbox(
        "Classifier Engine:",
        [
            "NeuroSwift Single Model (ECA + MultiScale, 86% Val Acc)",
            "NeuroSwift 5-Model Ensemble (90-94% Target Acc)",
        ],
    )

    st.header("📂 Input")
    st.info("Upload or select a real PhysioNet EEG (.edf) file for motor imagery classification.")

    source_mode = st.radio(
        "Select Input Source:",
        ["Upload EDF file", "Select from downloaded PhysioNet recordings"],
    )

    if source_mode == "Upload EDF file":
        uploaded_file = st.file_uploader("Upload PhysioNet EDF file", type=["edf", "EDF"])
        if uploaded_file is not None:
            with st.spinner("Processing EDF file..."):
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = tmp.name

                    raw = mne.io.read_raw_edf(tmp_path, preload=True, verbose="ERROR")
                    os.unlink(tmp_path)

                    X, y = process_raw_to_trials(raw, uploaded_file.name)
                    st.session_state.X = X
                    st.session_state.y = y
                    st.session_state.data_loaded = True
                    st.success(f"Loaded {len(X)} trials from {uploaded_file.name}!")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

    elif source_mode == "Select from downloaded PhysioNet recordings":
        data_raw_dir = ROOT / "data" / "raw"
        bundled_files = sorted(data_raw_dir.glob("*.edf")) if data_raw_dir.exists() else []
        sample_path = ROOT / "S001R01.edf"
        if sample_path.exists() and sample_path not in bundled_files:
            bundled_files.insert(0, sample_path)

        if bundled_files:
            file_names = [f.name for f in bundled_files]
            # Prioritize motor imagery runs
            default_idx = 0
            for i, fn in enumerate(file_names):
                if "R04" in fn or "R06" in fn or "R08" in fn:
                    default_idx = i
                    break

            selected_file_name = st.selectbox("Choose a PhysioNet recording:", file_names, index=default_idx)
            selected_file_path = next(f for f in bundled_files if f.name == selected_file_name)

            if st.button("Load Selected Recording", type="primary"):
                with st.spinner(f"Loading {selected_file_name}..."):
                    try:
                        raw = mne.io.read_raw_edf(str(selected_file_path), preload=True, verbose="ERROR")
                        X, y = process_raw_to_trials(raw, selected_file_name)
                        st.session_state.X = X
                        st.session_state.y = y
                        st.session_state.data_loaded = True
                        st.success(f"Loaded {len(X)} trials from {selected_file_name}!")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        else:
            st.warning("No downloaded EDF files found. Please upload an EDF file.")

    st.divider()
    if loaded_path:
        st.caption(f"**Loaded Weights:** `{Path(loaded_path).name}`")
    else:
        st.caption("No weights file found. Initialized fresh model.")

# Main content
if st.session_state.get("data_loaded", False) and "X" in st.session_state and len(st.session_state.X) > 0:
    X = st.session_state.X
    y = st.session_state.y

    if len(X) > 1:
        idx = st.slider("Select Trial Index", 0, len(X) - 1, 0)
    else:
        idx = 0
        st.caption("1 trial available")

    col1, col2 = st.columns(2)
    class_names = ["Left Hand", "Right Hand", "Both Hands", "Feet", "Rest"]

    with col1:
        st.metric("Total Trials Loaded", len(X))
        st.metric("Current Trial", f"{idx + 1} of {len(X)}")
    with col2:
        true_label = class_names[y[idx]] if (idx < len(y) and 0 <= y[idx] < len(class_names)) else "Rest / Baseline"
        st.metric("Ground Truth Label", true_label)

    # EEG plot
    st.subheader("🧠 Motor-Cortex Channels (C3, Cz, C4)")
    fig, ax = plt.subplots(figsize=(10, 4))
    channels = [8, 10, 12]  # C3, Cz, C4 indices
    channel_names = ["C3 (Left Motor)", "Cz (Foot Motor)", "C4 (Right Motor)"]
    t = np.arange(X.shape[-1]) / 160.0  # Time in seconds at 160 Hz
    for ch, name in zip(channels, channel_names):
        if ch < X.shape[1]:
            ax.plot(t, X[idx, ch], label=name, linewidth=1.2)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Amplitude (z-score)")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.5)
    st.pyplot(fig)
    plt.close(fig)

    if st.button("🔮 Classify Trial", use_container_width=True, type="primary"):
        use_ensemble = "Ensemble" in model_choice and ensemble_model is not None
        trial_norm = normalize_trial(X[idx : idx + 1])

        if use_ensemble:
            probs = ensemble_model.predict_proba(trial_norm)
            pred_idx = int(np.argmax(probs[0]))
            st.session_state.pred = class_names[pred_idx]
            st.session_state.conf = float(probs[0][pred_idx] * 100)
            st.session_state.all_probs = probs[0] * 100
            st.session_state.has_result = True
        elif single_model is not None:
            with torch.no_grad():
                input_data = torch.FloatTensor(trial_norm)
                outputs = single_model(input_data)
                probs = torch.softmax(outputs, dim=1)
                pred = torch.argmax(outputs, dim=1)

                st.session_state.pred = class_names[pred.item()]
                st.session_state.conf = float(probs[0][pred].item() * 100)
                st.session_state.all_probs = probs[0].numpy() * 100
                st.session_state.has_result = True
        else:
            st.warning("Model not loaded.")

    if st.session_state.get("has_result", False):
        st.subheader("🎯 Prediction Result")
        col1, col2 = st.columns([2, 1])
        with col1:
            st.success(f"**Predicted Class: {st.session_state.pred}**")
        with col2:
            st.metric("Model Confidence", f"{st.session_state.conf:.1f}%")

        st.write("#### Class Probabilities:")
        for i, name in enumerate(class_names):
            st.progress(float(st.session_state.all_probs[i] / 100), text=f"{name}: {st.session_state.all_probs[i]:.1f}%")

else:
    st.info("👈 Please upload an EDF file or select a recording from the sidebar to begin.")

    with st.expander("📖 How to use this app"):
        st.markdown(
            """
        1. **Select an EEG recording** from the sidebar (or upload any PhysioNet `.edf` file).
        2. Scrub through the trials using the **Trial Index slider**.
        3. Observe the filtered motor cortex EEG channels (**C3**, **Cz**, **C4**).
        4. Click **🔮 Classify Trial** to view the model's prediction, confidence percentage, and full probability distribution.
        
        **Dataset:** PhysioNet EEG Motor Movement/Imagery Dataset (EEGMMIDB, 64 channels, 160 Hz)  
        **Target Classes:** Left Hand, Right Hand, Both Hands, Feet, Rest
        """
        )
