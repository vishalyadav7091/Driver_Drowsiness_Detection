"""Streamlit dashboard for the Driver Drowsiness Detection System.

Run:  streamlit run app.py
"""
import tempfile
import threading
import time
from dataclasses import asdict, replace
from pathlib import Path

import cv2
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from src.alarm import Alarm, beep_html
from src.config import Config
from src.detector import DROWSY, SLEEPING
from src.video_pipeline import DrowsinessPipeline

st.set_page_config(page_title="Driver Drowsiness Detection", page_icon="\U0001F697", layout="wide")

STATE_HEX = {"ALERT": "#4A9B6B", "DROWSY": "#F2A900", "SLEEPING": "#D2493F",
             "NO FACE": "#6B7380", "CALIBRATING": "#3C9AC8"}

st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1250px;}
h1, h2, h3 {letter-spacing: -0.01em;}
.badge {display:inline-block; padding: .45rem 1.2rem; border-radius: 999px; font-weight: 700;
        font-size: 1.15rem; color: #0F1217; margin-bottom: .6rem;}
.tile {background:#1A1F27; border-radius:14px; padding:.9rem 1.1rem; margin-bottom:.7rem;}
.tile .lab {color:#8F9AAB; font-size:.78rem; letter-spacing:.08em; text-transform:uppercase;}
.tile .val {font-size:1.7rem; font-weight:700; color:#E6E9EE;}
.stTabs [data-baseweb="tab"] {font-weight:600;}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------- helpers
@st.cache_resource(show_spinner="Loading eye-state CNN...")
def get_eye_model():
    from src.eye_cnn import load_eye_model
    return load_eye_model()


def badge(state: str) -> str:
    return f'<span class="badge" style="background:{STATE_HEX.get(state, "#6B7380")}">{state}</span>'


def tile(label: str, value: str) -> str:
    return f'<div class="tile"><div class="lab">{label}</div><div class="val">{value}</div></div>'


def render_metrics(slot, res):
    slot.markdown(
        tile("EAR", f"{res.ear_smooth:.2f}") + tile("MAR", f"{res.mar:.2f}") + tile("PERCLOS", f"{res.perclos:.2f}")
        + tile("Blinks / Yawns / Nods", f"{res.blinks} / {res.yawns} / {res.nods}"),
        unsafe_allow_html=True)


# ---------------------------------------------------------------- sidebar
st.sidebar.title("Settings")
mode = st.sidebar.radio("Mode", ["Upload video", "Live webcam", "About"], index=0)
ear_thresh = st.sidebar.slider("EAR threshold", 0.10, 0.35, 0.21, 0.01)
closure_s = st.sidebar.slider("Closure time (s)", 0.5, 4.0, 1.5, 0.1)
model = get_eye_model()
use_cnn = st.sidebar.checkbox("Use eye-state CNN", value=model is not None, disabled=model is None)
if model is None:
    st.sidebar.caption("No CNN found at models/eye_state_cnn.keras, running on EAR only.")
alarm_on = st.sidebar.checkbox("Sound alarm (live mode)", value=True)
calibrate = st.sidebar.checkbox("Calibrate at start (5 s, look ahead)", value=False)
cfg = replace(Config(), EAR_THRESH=ear_thresh, CLOSURE_SECONDS=closure_s)
eye_model = model if use_cnn else None

st.title("Driver Drowsiness Detection")
st.caption("EAR landmarks + CNN eye state + temporal fusion, built with OpenCV, MediaPipe and Keras")


# ---------------------------------------------------------------- upload mode
if mode == "Upload video":
    up = st.file_uploader(
        "Upload a driving or face video",
        type=["mp4", "avi", "mov", "mkv"],
    )

    if up is not None and st.button("Analyse video", type="primary"):
        tmp_path = None
        cap = None
        pipe = None

        try:
            suffix = Path(up.name).suffix or ".mp4"

            with tempfile.NamedTemporaryFile(
                delete=False, suffix=suffix
            ) as tmp:
                tmp.write(up.getvalue())
                tmp_path = Path(tmp.name)

            cap = cv2.VideoCapture(str(tmp_path))

            if not cap.isOpened():
                st.error("Could not open the video. Try an MP4 video.")
            else:
                fps = cap.get(cv2.CAP_PROP_FPS)
                if not fps or fps <= 0:
                    fps = 30.0

                total = max(
                    1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                )

                pipe = DrowsinessPipeline(
                    cfg, eye_model, calibrate
                )

                progress = st.progress(0)
                status = st.empty()
                preview = st.empty()

                frame_number = 0

                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break

                    # Keep your existing pipeline API here.
                    # This call assumes process(frame) returns a result.
                    res = pipe.process(frame)

                    frame_number += 1
                    progress.progress(
                        min(1.0, frame_number / total)
                    )
                    status.write(
                        f"Analysing frame {frame_number} of {total}"
                    )

                    # Display the original frame until the
                    # pipeline's annotated-frame API is confirmed.
                    preview.image(
                        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                        channels="RGB",
                        use_container_width=True,
                    )

                status.success("Video analysis completed.")

        except Exception as exc:
            st.error(f"Video analysis failed: {exc}")

        finally:
            if cap is not None:
                cap.release()

            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)
