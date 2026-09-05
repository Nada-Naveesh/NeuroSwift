"""Reusable Streamlit UI fragments."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src.config import CLASS_NAMES, CHANNEL_NAMES, SAMPLING_RATE


DISCLAIMER = (
    "Academic research demo only. Predictions are probabilistic, not clinically validated, "
    "and must not be used for diagnosis or treatment decisions."
)


def inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');
        html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
        .block-container { padding-top: 1.4rem; max-width: 1200px; }
        .ns-hero {
            background: linear-gradient(135deg, #0b1f33 0%, #12344d 55%, #0e4d5c 100%);
            border: 1px solid #1f6f82;
            border-radius: 16px;
            padding: 1.4rem 1.6rem 1.2rem;
            margin-bottom: 1.1rem;
            color: #e8f4f8;
        }
        .ns-hero h1 { margin: 0 0 0.35rem 0; font-size: 1.85rem; letter-spacing: -0.02em; }
        .ns-hero p { margin: 0; color: #b7d3de; }
        .ns-chip {
            display: inline-block;
            background: rgba(56, 189, 176, 0.15);
            color: #7ee0d4;
            border: 1px solid rgba(56, 189, 176, 0.35);
            border-radius: 999px;
            padding: 0.15rem 0.65rem;
            font-size: 0.78rem;
            margin-right: 0.4rem;
        }
        .ns-pred {
            background: #102a3a;
            border-radius: 14px;
            border: 1px solid #2a6170;
            padding: 1.1rem 1.2rem;
        }
        .ns-pred .label { font-size: 1.6rem; font-weight: 700; color: #7ee0d4; }
        .ns-muted { color: #8aa4b3; font-size: 0.88rem; }
        footer { visibility: hidden; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero() -> None:
    st.markdown(
        """
        <div class="ns-hero">
          <span class="ns-chip">PhysioNet EEGMMIDB</span>
          <span class="ns-chip">5-class motor imagery</span>
          <span class="ns-chip">CNN + SE attention</span>
          <h1>NeuroSwift</h1>
          <p>Decode imagined Left Hand, Right Hand, Both Hands, Feet, or Rest from 64-channel EEG.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def probability_bars(probabilities: dict[str, float]) -> go.Figure:
    labels = list(CLASS_NAMES)
    values = [probabilities[k] * 100 for k in labels]
    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker=dict(color=["#38bdb0"] * 5),
            text=[f"{v:.1f}%" for v in values],
            textposition="outside",
        )
    )
    fig.update_layout(
        height=280,
        margin=dict(l=10, r=40, t=10, b=10),
        xaxis=dict(range=[0, 100], title="Probability (%)", gridcolor="#1d3a4a"),
        yaxis=dict(autorange="reversed"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#d5e6ee"),
        showlegend=False,
    )
    return fig


def eeg_plot(trial: np.ndarray, channels: tuple[int, ...] = (8, 10, 12)) -> go.Figure:
    n_times = trial.shape[-1]
    t = np.arange(n_times) / SAMPLING_RATE
    fig = go.Figure()
    names = []
    for i, ch in enumerate(channels):
        name = CHANNEL_NAMES[ch] if ch < len(CHANNEL_NAMES) else f"Ch{ch}"
        names.append(name)
        offset = (len(channels) - 1 - i) * 6.0
        fig.add_trace(
            go.Scatter(x=t, y=trial[ch] + offset, mode="lines", name=name, line=dict(width=1.4))
        )
    fig.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=30, b=10),
        title="Motor-cortex channels (C3, Cz, C4)",
        xaxis_title="Time (s)",
        yaxis_title="Amplitude (z-score, offset)",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#d5e6ee"),
        legend=dict(orientation="h"),
    )
    return fig
