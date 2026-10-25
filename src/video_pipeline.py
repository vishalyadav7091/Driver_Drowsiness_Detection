"""Frame pipeline: landmarks -> features -> CNN -> detector -> annotated frame.

CLI:
    python -m src.video_pipeline --video data/sample.mp4 --out outputs/annotated.mp4
    python -m src.video_pipeline --webcam
"""
import argparse
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import pandas as pd

from .alarm import Alarm
from .config import Config
from .detector import DrowsinessDetector
from .face_landmarks import FaceMeshDetector
from .features import LEFT_EYE, RIGHT_EYE, crop_eye, eye_aspect_ratio, head_pose, mouth_aspect_ratio
from .utils import FPSCounter, draw_eye, draw_hud


class DrowsinessPipeline:
    def __init__(self, cfg: Optional[Config] = None, eye_model=None, calibrate: bool = False):
        self.cfg = cfg or Config()
        self.mesh = FaceMeshDetector()
        self.eye_model = eye_model
        self.detector = DrowsinessDetector(self.cfg)
        self.fps = FPSCounter()
        self._i = 0
        self._last_cnn: Optional[float] = None
        if calibrate:
            self.detector.start_calibration()

    def process_frame(self, frame: np.ndarray, t: float):
        c = self.cfg
        h0, w0 = frame.shape[:2]
        if c.RESIZE_WIDTH and w0 > c.RESIZE_WIDTH:
            s = c.RESIZE_WIDTH / w0
            frame = cv2.resize(frame, (c.RESIZE_WIDTH, int(h0 * s)), interpolation=cv2.INTER_AREA)
        h, w = frame.shape[:2]
        self._i += 1

        lm = self.mesh.process(frame)
        if lm is None:
            self._last_cnn = None
            res = self.detector.update(t, False)
        else:
            left, right = lm[LEFT_EYE], lm[RIGHT_EYE]
            ear = (eye_aspect_ratio(left) + eye_aspect_ratio(right)) / 2
            mar = mouth_aspect_ratio(lm)
            pose = head_pose(lm, w, h)
            if self.eye_model is not None and self._i % c.CNN_EVERY == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                left_crop = crop_eye(gray, left, c.EYE_SIZE)
                right_crop = crop_eye(gray, right, c.EYE_SIZE)
                crops = [crop for crop in (left_crop, cv2.flip(right_crop, 1) if right_crop is not None else None)
                         if crop is not None]
                if crops:
                    self._last_cnn = self.eye_model.closed_prob(crops)
            res = self.detector.update(t, True, ear, mar, pose[0] if pose else None, self._last_cnn)
            col = (0, 169, 242)
            draw_eye(frame, left, col)
            draw_eye(frame, right, col)

        draw_hud(frame, res, self.fps.tick())
        return frame, res

    def close(self) -> None:
        self.mesh.close()


def run_video(path_in, path_out=None, csv_path=None, cfg: Optional[Config] = None, eye_model=None,
              progress: Optional[Callable[[int, int], None]] = None) -> pd.DataFrame:
    cfg = cfg or Config()
    cap = cv2.VideoCapture(str(path_in))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path_in}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    pipe = DrowsinessPipeline(cfg, eye_model)
    writer, rows, i = None, [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        out, res = pipe.process_frame(frame, i / fps)
        rows.append(asdict(res))
        if path_out:
            if writer is None:
                Path(path_out).parent.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(str(path_out), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                         (out.shape[1], out.shape[0]))
            writer.write(out)
        i += 1
        if progress and (i % 10 == 0 or i == total):
            progress(i, total)
    cap.release()
    if writer:
        writer.release()
    pipe.close()
    df = pd.DataFrame(rows)
    if csv_path:
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
    return df


def run_webcam(cam: int = 0, cfg: Optional[Config] = None, eye_model=None, calibrate: bool = False) -> None:
    cfg = cfg or Config()
    cap = cv2.VideoCapture(cam)
    pipe, alarm, t0 = DrowsinessPipeline(cfg, eye_model, calibrate), Alarm(cfg), time.time()
    print("Press q to quit.")
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        t = time.time() - t0
        out, res = pipe.process_frame(frame, t)
        if alarm.should_sound(res.state, t):
            alarm.play_local()
        cv2.imshow("Driver Drowsiness Detection", out)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
    pipe.close()


if __name__ == "__main__":
    from .eye_cnn import load_eye_model

    ap = argparse.ArgumentParser()
    ap.add_argument("--video")
    ap.add_argument("--out", default="outputs/annotated.mp4")
    ap.add_argument("--csv", default="outputs/log.csv")
    ap.add_argument("--webcam", action="store_true")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--no-cnn", action="store_true")
    a = ap.parse_args()
    model = None if a.no_cnn else load_eye_model()
    print("CNN:", "loaded" if model else "not used (EAR only)")
    if a.webcam:
        run_webcam(eye_model=model, calibrate=a.calibrate)
    elif a.video:
        df = run_video(a.video, a.out, a.csv, eye_model=model, progress=lambda i, n: print(f"\r{i}/{n}", end=""))
        print("\n", df["state"].value_counts().to_string())
    else:
        ap.error("give --video PATH or --webcam")
