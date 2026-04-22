

from __future__ import annotations

import sys
import time
import json
import logging
from pathlib import Path
from collections import deque

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    SMAP_CHANNEL,
    STREAM_INTERVAL_SEC,
    ALERT_LOG_PATH,
    MODELS_DIR,
)
from src.simulator import SensorStreamSimulator
from src.detector import AnomalyDetector

logging.basicConfig(level=logging.WARNING)

st.set_page_config(
    page_title="Anomaly Detection | NASA SMAP",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("", unsafe_allow_html=True)

def init_state():
    defaults = {
        "running": False,
        "timestamps": deque(maxlen=300),
        "signal_values": deque(maxlen=300),
        "anomaly_flags": deque(maxlen=300),
        "ensemble_scores": deque(maxlen=300),
        "if_scores": deque(maxlen=300),
        "ocsvm_scores": deque(maxlen=300),
        "lstm_scores": deque(maxlen=300),
        "alerts": [],
        "last_result": None,
        "tick": 0,
        "total_windows": 0,
        "total_anomalies": 0,
        "simulator": None,
        "detector": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

with st.sidebar:
    st.title("Control Panel")
    st.markdown("---")

    entity = st.selectbox("SMAP Entity", [SMAP_CHANNEL, "S-1", "E-1", "A-1"], index=0)
    interval = st.slider("Stream interval (sec)", 0.5, 5.0, float(STREAM_INTERVAL_SEC), 0.5)
    run_shap = st.toggle("SHAP Explanations", value=False)
    show_channels = st.toggle("Show all channels", value=False)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        start_btn = st.button("Start", type="primary", use_container_width=True)
    with col2:
        stop_btn = st.button("Stop", use_container_width=True)

    if start_btn:
        st.session_state.running = True

        models_exist = (MODELS_DIR / f"{entity}_isolation_forest.joblib").exists()
        if not models_exist:
            st.error(f"Models not found for entity '{entity}'. Run: python -m src.train --entity {entity}")
            st.session_state.running = False
        else:
            st.session_state.simulator = SensorStreamSimulator(entity, interval_sec=interval)
            st.session_state.detector = AnomalyDetector(entity)

            for key in ["timestamps", "values", "anomaly_flags", "ensemble_scores",
                        "if_scores", "ocsvm_scores", "lstm_scores"]:
                st.session_state[key] = deque(maxlen=300)
            st.session_state.alerts = []
            st.session_state.tick = 0
            st.session_state.total_windows = 0
            st.session_state.total_anomalies = 0

    if stop_btn:
        st.session_state.running = False

    st.markdown("---")
    st.markdown("**Models**")
    st.markdown("- Isolation Forest")
    st.markdown("- One-Class SVM")
    st.markdown("- LSTM Autoencoder (PyTorch)")
    st.markdown("**Dataset**")
    st.markdown("- NASA SMAP sensor telemetry")
    st.markdown("**Ensemble**")
    st.markdown("- Majority voting (2/3)")

st.title("Real-time Anomaly Detection")
st.markdown("**NASA SMAP Sensor Data | Ensemble: Isolation Forest + One-Class SVM + LSTM Autoencoder**")

metric_cols = st.columns(5)
with metric_cols[0]:
    status_html = ('<span class="anomaly-badge">ANOMALY</span>'
                   if st.session_state.last_result and st.session_state.last_result.is_anomaly
                   else '<span class="normal-badge">NORMAL</span>')
    st.markdown(f"**Status**<br>{status_html}", unsafe_allow_html=True)

with metric_cols[1]:
    st.metric("Total Windows", st.session_state.total_windows)

with metric_cols[2]:
    st.metric("Anomalies Detected", st.session_state.total_anomalies)

with metric_cols[3]:
    rate = (st.session_state.total_anomalies / max(st.session_state.total_windows, 1)) * 100
    st.metric("Anomaly Rate", f"{rate:.1f}%")

with metric_cols[4]:
    last_score = list(st.session_state.ensemble_scores)[-1] if st.session_state.ensemble_scores else 0
    st.metric("Ensemble Score", f"{last_score:.3f}")

st.markdown("---")

chart_placeholder = st.empty()
score_chart_placeholder = st.empty()

col_shap, col_alerts = st.columns([1, 1])
with col_shap:
    shap_placeholder = st.empty()
with col_alerts:
    alert_placeholder = st.empty()

def render_main_chart():
    ts = list(st.session_state.timestamps)
    vals = list(st.session_state.signal_values)
    flags = list(st.session_state.anomaly_flags)

    if not ts:
        return

    df = pd.DataFrame({"timestamp": ts, "value": vals, "anomaly": flags})
    normal = df[df["anomaly"] == 0]
    anomalous = df[df["anomaly"] == 1]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=normal.index, y=normal["value"],
        mode="lines", name="Normal",
        line=dict(color="#7eb8f7", width=1.5),
    ))
    if not anomalous.empty:
        fig.add_trace(go.Scatter(
            x=anomalous.index, y=anomalous["value"],
            mode="markers", name="Anomaly",
            marker=dict(color="#ff4b4b", size=8, symbol="x"),
        ))
    fig.update_layout(
        title="Primary Telemetry Channel",
        xaxis_title="Tick",
        yaxis_title="Normalized Value",
        height=300,
        margin=dict(l=40, r=20, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font=dict(color="#fafafa"),
    )
    chart_placeholder.plotly_chart(fig, use_container_width=True, key="main_chart")

def render_score_chart():
    if_scores = list(st.session_state.if_scores)
    ocsvm_scores = list(st.session_state.ocsvm_scores)
    lstm_scores = list(st.session_state.lstm_scores)
    ensemble_scores = list(st.session_state.ensemble_scores)

    if not ensemble_scores:
        return

    fig = go.Figure()
    x = list(range(len(ensemble_scores)))
    for name, scores, color in [
        ("Ensemble", ensemble_scores, "#ffd700"),
        ("LSTM AE", lstm_scores, "#ff6b6b"),
        ("Isolation Forest", if_scores, "#7eb8f7"),
        ("One-Class SVM", ocsvm_scores, "#98d98e"),
    ]:
        fig.add_trace(go.Scatter(
            x=x, y=scores, mode="lines", name=name,
            line=dict(color=color, width=1.5 if name != "Ensemble" else 2.5),
        ))

    fig.add_hline(y=1.0, line_dash="dash", line_color="red", annotation_text="Threshold")

    fig.update_layout(
        title="Anomaly Scores (normalized to threshold)",
        xaxis_title="Tick",
        yaxis_title="Score / Threshold",
        height=250,
        margin=dict(l=40, r=20, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font=dict(color="#fafafa"),
    )
    score_chart_placeholder.plotly_chart(fig, use_container_width=True, key="score_chart")

def render_shap(result):
    if not result or not result.top_features:
        shap_placeholder.info("SHAP: Enable 'SHAP Explanations' in sidebar and wait for anomaly")
        return

    features = result.top_features
    df = pd.DataFrame(features)
    df["color"] = df["direction"].map({"up": "#ff4b4b", "down": "#7eb8f7"})
    df = df.sort_values("shap_value", key=abs, ascending=True)

    fig = go.Figure(go.Bar(
        x=df["shap_value"],
        y=df["feature"],
        orientation="h",
        marker_color=df["color"],
    ))
    fig.update_layout(
        title="SHAP — Why is this an anomaly?",
        xaxis_title="SHAP Value",
        height=250,
        margin=dict(l=140, r=20, t=40, b=40),
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font=dict(color="#fafafa"),
    )
    shap_placeholder.plotly_chart(fig, use_container_width=True, key="shap_chart")

def render_alerts():
    if not st.session_state.alerts:
        alert_placeholder.info("No anomalies detected yet")
        return

    alerts_df = pd.DataFrame(st.session_state.alerts[-10:][::-1])
    cols_to_show = ["timestamp", "ensemble_score", "votes", "if_pred", "ocsvm_pred", "lstm_pred"]
    cols_to_show = [c for c in cols_to_show if c in alerts_df.columns]
    alert_placeholder.markdown("**Recent Alerts**")
    alert_placeholder.dataframe(alerts_df[cols_to_show], use_container_width=True)

if st.session_state.running and st.session_state.simulator and st.session_state.detector:
    sim = st.session_state.simulator
    det = st.session_state.detector

    try:
        window, meta = next(iter(sim.stream()))
        result = det.predict(window, run_shap=run_shap)

        st.session_state.timestamps.append(meta["timestamp"])
        st.session_state.signal_values.append(meta["primary_channel_last"])
        st.session_state.anomaly_flags.append(int(result.is_anomaly))
        st.session_state.ensemble_scores.append(result.ensemble_score)
        st.session_state.if_scores.append(result.if_score / (det.if_detector.threshold + 1e-10))
        st.session_state.ocsvm_scores.append(result.ocsvm_score / (det.ocsvm_detector.threshold + 1e-10))
        st.session_state.lstm_scores.append(result.lstm_score / (det.lstm_trainer.threshold + 1e-10))
        st.session_state.total_windows += 1
        if result.is_anomaly:
            st.session_state.total_anomalies += 1
            st.session_state.alerts.append(result.__dict__)
        st.session_state.last_result = result
        st.session_state.tick += 1

    except StopIteration:
        st.session_state.running = False

render_main_chart()
render_score_chart()
render_shap(st.session_state.last_result)
render_alerts()

if st.session_state.running:
    time.sleep(interval)
    st.rerun()
elif not st.session_state.running and st.session_state.total_windows == 0:
    st.info("Click **Start** in the sidebar to begin the real-time stream.")


