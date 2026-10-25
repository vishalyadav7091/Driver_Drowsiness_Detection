"""Thin MediaPipe Face Mesh wrapper returning pixel landmarks."""
from typing import Optional

import cv2
import numpy as np


class FaceMeshDetector:
    def __init__(self, refine: bool = True, det_conf: float = 0.5, track_conf: float = 0.5):
        import mediapipe as mp  # imported lazily so tests run without it

        self._mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=refine,
            min_detection_confidence=det_conf,
            min_tracking_confidence=track_conf,
        )

    def process(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Return (N, 2) float32 pixel landmarks, or None if no face."""
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        res = self._mesh.process(rgb)
        if not res.multi_face_landmarks:
            return None
        pts = res.multi_face_landmarks[0].landmark
        return np.array([(p.x * w, p.y * h) for p in pts], dtype=np.float32)

    def close(self) -> None:
        self._mesh.close()
