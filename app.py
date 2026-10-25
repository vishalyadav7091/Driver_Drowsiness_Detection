
"""Driver Drowsiness Detection - Streamlit dashboard.

Run:
    streamlit run app.py
"""

import tempfile
import time
from dataclasses import replace
from pathlib import Path

import cv2
import streamlit as st
import streamlit.components.v1 as components

from src.alarm import Alarm, beep_html
from src.config import Config
from src.detector import SLEEPING
from src.video_pipeline import DrowsinessPipeline


st.set_page_config(
    page_title="Driver Drowsiness Detection",
    page_icon="🚗",
    layout="wide",
)

STATE_HEX = {
    "ALERT": "#4A9B6B",
    "DROWSY": "#F2A900",
    "SLEEPING": "#D2493F",
    "NO FACE": "#6B7380",
    "CALIBRATING": "#3C9AC8",
}

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; max-width: 1250px;}
    h1, h2, h3 {letter-spacing: -0.01em;}
    .badge {
        display:inline-block; padding:.45rem 1.2rem;
        border-radius:999px; font-weight:700;
        font-size:1.15rem; color:#0F1217;
        margin-bottom:.6rem;
    }
    .tile {
        background:#1A1F27; border-radius:14px;
        padding:.9rem 1.1rem; margin-bottom:.7rem;
    }
    .tile .lab {
        color:#8F9AAB; font-size:.78rem;
        letter-spacing:.08em; text-transform:uppercase;
    }
    .tile .val {
        font-size:1.7rem; font-weight:700; color:#E6E9EE;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def badge(state):
    color = STATE_HEX.get(state, "#6B7380")
    return (
        f'<span class="badge" style="background:{color}">'
        f"{state}</span>"
    )


def tile(label, value):
    return (
        f'<div class="tile"><div class="lab">{label}</div>'
        f'<div class="val">{value}</div></div>'
    )


def render_metrics(slot, result):
    slot.markdown(
        tile("EAR", f"{result.ear_smooth:.2f}")
        + tile("MAR", f"{result.mar:.2f}")
        + tile("PERCLOS", f"{result.perclos:.2f}")
        + tile(
            "Blinks / Yawns / Nods",
            f"{result.blinks} / {result.yawns} / {result.nods}",
        ),
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner="Loading eye-state CNN...")
def get_eye_model():
    try:
        from src.eye_cnn import load_eye_model
        return load_eye_model()
    except (ImportError, FileNotFoundError, OSError, ValueError) as exc:
        return None


st.sidebar.title("Settings")

mode = st.sidebar.radio(
    "Mode",
    ["Upload video", "Live webcam", "About"],
)

ear_thresh = st.sidebar.slider(
    "EAR threshold", 0.10, 0.35, 0.21, 0.01
)
closure_s = st.sidebar.slider(
    "Closure time (s)", 0.5, 4.0, 1.5, 0.1
)

model = get_eye_model()
use_cnn = st.sidebar.checkbox(
    "Use eye-state CNN",
    value=model is not None,
    disabled=model is None,
)

if model is None:
    st.sidebar.caption(
        "CNN unavailable. The application will attempt to use EAR only."
    )

alarm_on = st.sidebar.checkbox(
    "Sound alarm", value=True
)
calibrate = st.sidebar.checkbox(
    "Calibrate at start (5 seconds)", value=False
)

cfg = replace(
    Config(),
    EAR_THRESH=ear_thresh,
    CLOSURE_SECONDS=closure_s,
)
eye_model = model if use_cnn else None

st.title("🚗 Driver Drowsiness Detection")
st.caption(
    "Eye Aspect Ratio + CNN eye state + temporal analysis"
)


# ----------------------------------------------------------
# Upload video
# ----------------------------------------------------------
if mode == "Upload video":
    st.subheader("Video analysis")

    uploaded = st.file_uploader(
        "Upload a driving or face video",
        type=["mp4", "avi", "mov", "mkv"],
    )

    if uploaded is not None and st.button(
        "Analyse video", type="primary"
    ):
        tmp_path = None
        cap = None
        pipeline = None

        try:
            suffix = Path(uploaded.name).suffix or ".mp4"

            with tempfile.NamedTemporaryFile(
                delete=False, suffix=suffix
            ) as tmp:
                tmp.write(uploaded.getvalue())
                tmp_path = Path(tmp.name)

            cap = cv2.VideoCapture(str(tmp_path))

            if not cap.isOpened():
                st.error(
                    "Could not open the video. Try an MP4 file."
                )
            else:
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps <= 0:
                    fps = 30.0

                total = max(
                    1,
                    int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                )

                pipeline = DrowsinessPipeline(
                    cfg, eye_model, calibrate
                )

                progress = st.progress(0)
                status = st.empty()
                preview = st.empty()
                metrics = st.container()

                frame_count = 0
                last_result = None

                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break

                    # This assumes your pipeline exposes process(frame).
                    result = pipeline.process(frame)
                    last_result = result

                    frame_count += 1

                    preview.image(
                        cv2.cvtColor(
                            frame, cv2.COLOR_BGR2RGB
                        ),
                        channels="RGB",
                        use_container_width=True,
                    )

                    progress.progress(
                        min(1.0, frame_count / total)
                    )
                    status.write(
                        f"Processed {frame_count} / {total} frames"
                    )

                if last_result is not None:
                    with metrics:
                        st.subheader("Last-frame metrics")
                        render_metrics(st, last_result)

                status.success(
                    f"Analysis finished. Processed {frame_count} frames."
                )

        except Exception as exc:
            st.error(f"Video analysis failed: {exc}")

        finally:
            if cap is not None:
                cap.release()

            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)


# ----------------------------------------------------------
# Live webcam
# ----------------------------------------------------------
elif mode == "Live webcam":
    st.subheader("Live monitoring")
    st.info(
        "Allow camera access in your browser. "
        "If the connection times out, the deployment may need TURN."
    )

    try:
        from streamlit_webrtc import (
            VideoProcessorBase,
            webrtc_streamer,
        )
        import av

        class DrowsinessVideoProcessor(VideoProcessorBase):
            def __init__(self):
                self.pipeline = DrowsinessPipeline(
                    cfg, eye_model, calibrate
                )
                self.alarm = Alarm(cfg)
                self.last_state = "CALIBRATING"
                self.last_result = None
                self.last_alarm_time = 0.0

            def recv(self, frame):
                image = frame.to_ndarray(format="bgr24")

                try:
                    result = self.pipeline.process(image)
                    self.last_result = result

                    state = getattr(
                        result, "state", "ALERT"
                    )
                    self.last_state = state

                    color = (
                        0, 0, 255
                    ) if state == SLEEPING else (0, 200, 0)

                    cv2.putText(
                        image,
                        f"State: {state}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        color,
                        2,
                    )

                except Exception as exc:
                    cv2.putText(
                        image,
                        "Detection error",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2,
                    )

                return av.VideoFrame.from_ndarray(
                    image, format="bgr24"
                )

        ctx = webrtc_streamer(
            key="drowsiness-monitor",
            video_processor_factory=DrowsinessVideoProcessor,
            rtc_configuration={
                "iceServers": [
                    {
                        "urls": [
                            "stun:stun.l.google.com:19302"
                        ]
                    }
                ]
            },
            media_stream_constraints={
                "video": True,
                "audio": False,
            },
            async_processing=True,
        )

        if ctx.state.playing and ctx.video_processor:
            processor = ctx.video_processor
            state = processor.last_state

            st.markdown(badge(state), unsafe_allow_html=True)

            if alarm_on and state == SLEEPING:
                now = time.monotonic()

                if now - processor.last_alarm_time >= cfg.ALARM_COOLDOWN_S:
                    processor.last_alarm_time = now
                    components.html(
                        beep_html(seconds=3.0, freq=1000),
                        height=60,
                    )

            if processor.last_result is not None:
                render_metrics(st, processor.last_result)

    except ImportError:
        st.error(
            "Install streamlit-webrtc and av in requirements.txt."
        )
    except Exception as exc:
        st.error(f"Live webcam initialization failed: {exc}")


# ----------------------------------------------------------
# About
# ----------------------------------------------------------
else:
    st.subheader("How it works")

    st.markdown(
        """
        1. **Face landmarks:** facial landmarks are detected from video frames.
        2. **Eye analysis:** EAR estimates whether the eyes are closed.
        3. **CNN classification:** when available, a trained model classifies eye state.
        4. **Temporal analysis:** the pipeline evaluates drowsiness over time.
        5. **Warning:** an alarm can warn when the system detects sleeping.
        """
    )

    st.warning(
        "This project is a prototype, not a certified driver-safety device. "
        "Do not rely on it as a substitute for rest or safe driving."
    )
