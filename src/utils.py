"""Drawing helpers and an FPS counter."""
import time

import cv2
import numpy as np

STATE_BGR = {
    "ALERT": (107, 155, 74),
    "DROWSY": (0, 169, 242),
    "SLEEPING": (63, 73, 210),
    "NO FACE": (115, 128, 107),
    "CALIBRATING": (200, 160, 60),
}


class FPSCounter:
    def __init__(self, alpha: float = 0.9):
        self._last = None
        self.fps = 0.0
        self._a = alpha

    def tick(self) -> float:
        now = time.perf_counter()
        if self._last is not None:
            inst = 1.0 / max(now - self._last, 1e-6)
            self.fps = inst if self.fps == 0 else self._a * self.fps + (1 - self._a) * inst
        self._last = now
        return self.fps


def draw_eye(img, pts, color=(0, 169, 242)):
    cv2.polylines(img, [np.asarray(pts, np.int32).reshape(-1, 1, 2)], True, color, 1, cv2.LINE_AA)
    for p in np.asarray(pts, np.int32):
        cv2.circle(img, tuple(int(v) for v in p), 2, color, -1, cv2.LINE_AA)


def draw_hud(img, res, fps: float):
    h, w = img.shape[:2]
    col = STATE_BGR.get(res.state, (128, 128, 128))
    cv2.rectangle(img, (0, 0), (w, 42), col, -1)
    cv2.putText(img, res.state, (12, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, f"{fps:4.1f} FPS", (w - 110, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    if res.face:
        lines = [f"EAR {res.ear_smooth:.2f}", f"MAR {res.mar:.2f}", f"Pitch {res.pitch:+.0f}",
                 f"PERCLOS {res.perclos:.2f}", f"Blinks {res.blinks}  Yawns {res.yawns}  Nods {res.nods}"]
        if res.cnn_closed >= 0:
            lines.insert(1, f"CNN closed {res.cnn_closed:.2f}")
        for i, ln in enumerate(lines):
            cv2.putText(img, ln, (12, 70 + i * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    if res.state == "SLEEPING":
        cv2.rectangle(img, (0, 0), (w - 1, h - 1), col, 6)
    return img
