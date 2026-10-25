# Driver Drowsiness Detection

Real-time drowsiness detection from a camera or a video, using **MediaPipe Face Mesh**, the **Eye Aspect Ratio (EAR)**, a small **Keras CNN** for eye state, and a fusion state machine (`ALERT` / `DROWSY` / `SLEEPING`). Includes a Streamlit dashboard.

```
frame -> Face Mesh -> [ EAR, MAR, head pitch ] --+
                   -> eye crops -> CNN ----------+-> fusion + PERCLOS -> state -> alarm / dashboard / CSV
```

## Quick start (local)

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1. data: put images in data/eyes/closed and data/eyes/open
#    (MRL Eye Dataset:  python scripts/prepare_mrl.py PATH_TO_MRL --limit 15000)
# 2. train the eye-state CNN
python -m src.eye_cnn --data data/eyes --epochs 20          # -> models/eye_state_cnn.keras

# 3a. run on a video
python -m src.video_pipeline --video clip.mp4 --out outputs/annotated.mp4
# 3b. run on the webcam (press q to quit)
python -m src.video_pipeline --webcam --calibrate
# 3c. dashboard
streamlit run app.py
```

No CNN yet? Everything still runs on EAR alone (`--no-cnn`, or just skip training).

## Colab

Open `notebooks/Driver_Drowsiness_Detection.ipynb`, upload the repo zip, and run the cells in order. Colab has no webcam, so use an uploaded video.

## How the decision is made

| Signal | Rule |
|---|---|
| Closed eye | smoothed EAR below 0.21 **and** CNN agrees; or CNN above 0.85 with EAR borderline |
| SLEEPING | eyes closed for 1.5 s or more (timestamp based, so FPS independent) |
| DROWSY | PERCLOS above 0.15 over 45 s, or 2+ yawns or 2+ head nods in 60 s |
| Hysteresis | a state must hold 1 s before dropping to a lower state |

Every threshold is in `src/config.py`. `--calibrate` (or the sidebar checkbox) sets a personal EAR threshold from 5 s of open eyes.

## Layout

```
app.py                 Streamlit dashboard
src/config.py          all thresholds and paths
src/face_landmarks.py  MediaPipe wrapper
src/features.py        EAR, MAR, head pose, eye crop
src/eye_cnn.py         CNN build / train / evaluate / inference
src/dataset.py         stratified 70/15/15 split, tf.data pipeline
src/detector.py        fusion + state machine
src/video_pipeline.py  frame loop, annotated video, CSV log, webcam
src/alarm.py           generated beep, local + browser playback
scripts/prepare_mrl.py sort the MRL dataset into open/closed
tests/                 pytest unit tests
```

## Deploy on Streamlit Community Cloud

1. Push the repo to GitHub. Commit `models/eye_state_cnn.keras` (it is a few MB).
2. On share.streamlit.io choose the repo, main file `app.py`, **Python 3.11**.
3. `requirements.txt` and `packages.txt` are picked up automatically.
4. Upload mode works everywhere. Live webcam mode uses WebRTC and may need a TURN server on strict networks.

## Tests

```bash
pytest -q
```

## Limits

Low light, sunglasses and extreme head angles reduce accuracy. This is a student project, not a certified safety device.
