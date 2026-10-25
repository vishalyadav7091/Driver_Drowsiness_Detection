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
    up = st.file_uploader("Upload a driving or face video", type=["mp4", "avi", "mov", "mkv"])
    if up is not None and st.button("Analyse video", type="primary"):
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(up.name).suffix)
        tmp_path = Path(tmp.name)
        cap, pipe = None, None
        try:
            tmp.write(up.read())
            tmp.close()
            cap = cv2.VideoCapture(str(tmp_path))
            if not cap.isOpened():
                st.error("Could not open that video. Try an MP4 (H.264).")
                st.stop()
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
            pipe = DrowsinessPipeline(cfg, eye_model, calibrate)

            left, right = st.columns([3, 1.25])
            frame_slot, badge_slot, met_slot = left.empty(), right.empty(), right.empty()
            bar = st.progress(0.0)
            rows, i = [], 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                out, res = pipe.process_frame(frame, i / fps)
                rows.append(asdict(res))
                if i % 3 == 0:
                    frame_slot.image(out, channels="BGR", use_container_width=True)
                    badge_slot.markdown(badge(res.state), unsafe_allow_html=True)
                    render_metrics(met_slot, res)
                i += 1
                if i % 10 == 0:
                    bar.progress(min(i / total, 1.0))
            bar.progress(1.0)
        finally:
            try:
                tmp.close()
                if cap is not None:
                    cap.release()
                if pipe is not None:
                    pipe.close()
            finally:
                if tmp_path.exists():
                    tmp_path.unlink()
        if not rows:
            st.warning("No frames were read from the video.")
            st.stop()
        st.session_state["log"] = pd.DataFrame(rows)

    df = st.session_state.get("log")
    if df is not None and len(df):
        st.subheader("Results")
        counts = df["state"].value_counts(normalize=True).mul(100).round(1)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Duration", f"{df['t'].iloc[-1]:.1f} s")
        c2.metric("Time in SLEEPING", f"{counts.get(SLEEPING, 0):.1f} %")
        c3.metric("Time in DROWSY", f"{counts.get(DROWSY, 0):.1f} %")
        c4.metric("Blinks / yawns / nods", f"{df['blinks'].iloc[-1]} / {df['yawns'].iloc[-1]} / {df['nods'].iloc[-1]}")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["t"], y=df["ear_smooth"], name="EAR", line=dict(color="#F2A900", width=2)))
        fig.add_hline(y=ear_thresh, line_dash="dash", line_color="#8F9AAB", annotation_text="threshold")
        sleeping = df["state"] == SLEEPING
        if sleeping.any():
            grp = (sleeping != sleeping.shift()).cumsum()
            for _, g in df[sleeping].groupby(grp[sleeping]):
                fig.add_vrect(x0=g["t"].iloc[0], x1=g["t"].iloc[-1], fillcolor="#D2493F", opacity=0.25, line_width=0)
        fig.update_layout(template="plotly_dark", height=320, margin=dict(l=10, r=10, t=30, b=10),
                          title="Eye aspect ratio over time (red = SLEEPING)", xaxis_title="seconds")
        st.plotly_chart(fig, use_container_width=True)

        a, b = st.columns(2)
        dist = df["state"].value_counts().reset_index()
        dist.columns = ["state", "frames"]
        bar_fig = go.Figure(go.Bar(x=dist["state"], y=dist["frames"], marker_color=[STATE_HEX.get(s, "#6B7380") for s in dist["state"]]))
        bar_fig.update_layout(template="plotly_dark", height=280, title="State distribution", margin=dict(l=10, r=10, t=40, b=10))
        a.plotly_chart(bar_fig, use_container_width=True)

        change = df["state"] != df["state"].shift()
        events = df.loc[change, ["t", "state"]].rename(columns={"t": "time (s)"})
        events["time (s)"] = events["time (s)"].round(2)
        b.markdown("**State changes**")
        b.dataframe(events, hide_index=True, height=240, use_container_width=True)

        st.download_button("Download frame-by-frame CSV", df.to_csv(index=False).encode(),
                           file_name="drowsiness_log.csv", mime="text/csv")

# ---------------------------------------------------------------- live mode
elif mode == "Live webcam":
    try:
        import av
        from streamlit_webrtc import VideoProcessorBase, webrtc_streamer
    except Exception:
        st.error("Live mode needs `streamlit-webrtc` and `av`. Install with: pip install streamlit-webrtc av")
        st.stop()

    class Processor(VideoProcessorBase):
        def __init__(self):
            self.pipe = DrowsinessPipeline(cfg, eye_model, calibrate)
            self.t0 = time.time()
            self.lock = threading.Lock()
            self.latest = None

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            out, res = self.pipe.process_frame(img, time.time() - self.t0)
            with self.lock:
                self.latest = res
            return av.VideoFrame.from_ndarray(out, format="bgr24")

    st.info("Allow camera access, then press START. Changing settings in the sidebar needs a stop and start.")
    ctx = webrtc_streamer(
        key="drowsiness-live",
        video_processor_factory=Processor,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    badge_slot, met_slot, audio_slot = st.empty(), st.empty(), st.empty()
    alarm = Alarm(cfg)
    while ctx.state.playing:
        proc = ctx.video_processor
        res = None
        if proc is not None:
            with proc.lock:
                res = proc.latest
        if res is not None:
            badge_slot.markdown(badge(res.state), unsafe_allow_html=True)
            render_metrics(met_slot, res)
            if alarm_on and alarm.should_sound(res.state, time.time()):
                with audio_slot:
                    components.html(beep_html(), height=0)
        time.sleep(0.25)

# ---------------------------------------------------------------- about
else:
    st.subheader("How it works")
    st.markdown(f"""
1. **MediaPipe Face Mesh** finds 478 landmarks on every frame.
2. **Geometry**: the Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR) and head pitch are computed from landmarks.
3. **Eye CNN**: each eye is cropped and classified open or closed by a small Keras CNN.
4. **Fusion**: an eye counts as closed when EAR and the CNN agree, or when the CNN is above {cfg.CNN_CONF:.2f} and EAR is borderline.
5. **State machine**: `ALERT` to `DROWSY` (PERCLOS above {cfg.PERCLOS_DROWSY}, repeated yawns or nods) to `SLEEPING` (eyes closed for {cfg.CLOSURE_SECONDS} s or more), with hysteresis so the label does not flicker.

**Privacy**: live webcam frames are processed in memory only; an uploaded video is kept in a temporary file for analysis.
""")
    st.caption("Limits: low light, sunglasses and extreme head angles reduce accuracy. Not a certified safety device.")
