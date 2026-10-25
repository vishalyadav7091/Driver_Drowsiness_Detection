"""EAR, MAR, head pose and eye cropping from Face Mesh landmarks."""
from typing import Optional, Tuple

import numpy as np

# p1..p6 order: outer corner, upper x2, inner corner, lower x2
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH_CORNERS = (78, 308)
MOUTH_PAIRS = ((81, 178), (13, 14), (311, 402))


def _d(a, b) -> float:
    return float(np.linalg.norm(np.asarray(a, float) - np.asarray(b, float)))


def eye_aspect_ratio(p) -> float:
    """EAR = (|p2-p6| + |p3-p5|) / (2 |p1-p4|) for six eye points."""
    p = np.asarray(p, float)
    horiz = _d(p[0], p[3])
    if horiz < 1e-6:
        return 0.0
    return (_d(p[1], p[5]) + _d(p[2], p[4])) / (2.0 * horiz)


def mouth_aspect_ratio(lm) -> float:
    horiz = _d(lm[MOUTH_CORNERS[0]], lm[MOUTH_CORNERS[1]])
    if horiz < 1e-6:
        return 0.0
    vert = sum(_d(lm[a], lm[b]) for a, b in MOUTH_PAIRS)
    return vert / (3.0 * horiz)


_MODEL_3D = np.array([
    (0.0, 0.0, 0.0),          # nose tip (1)
    (0.0, -63.6, -12.5),      # chin (152)
    (-43.3, 32.7, -26.0),     # eye outer corner, image-left (33)
    (43.3, 32.7, -26.0),      # eye outer corner, image-right (263)
    (-28.9, -28.9, -24.1),    # mouth corner, image-left (61)
    (28.9, -28.9, -24.1),     # mouth corner, image-right (291)
], dtype=np.float64)
_POSE_IDX = [1, 152, 33, 263, 61, 291]


def head_pose(lm: np.ndarray, w: int, h: int) -> Optional[Tuple[float, float, float]]:
    """Return (pitch, yaw, roll) in degrees via solvePnP, or None on failure."""
    import cv2

    img_pts = np.array([lm[i] for i in _POSE_IDX], dtype=np.float64)
    focal = float(w)
    cam = np.array([[focal, 0, w / 2], [0, focal, h / 2], [0, 0, 1]], dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(_MODEL_3D, img_pts, cam, np.zeros((4, 1)), flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    rmat, _ = cv2.Rodrigues(rvec)
    angles = cv2.RQDecomp3x3(rmat)[0]
    pitch, yaw, roll = angles
    if pitch > 90:      # fold the +-180 wrap so a frontal face reads near 0
        pitch -= 180
    elif pitch < -90:
        pitch += 180
    if yaw > 90:
        yaw = 180 - yaw
    elif yaw < -90:
        yaw = -180 - yaw
    return float(pitch), float(yaw), float(roll)


def crop_eye(gray: np.ndarray, pts: np.ndarray, size: int = 64, scale: float = 1.6) -> Optional[np.ndarray]:
    """Square grayscale crop around an eye, resized to size x size."""
    import cv2

    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half = max(x2 - x1, y2 - y1) * scale / 2
    h, w = gray.shape[:2]
    xa, xb = int(max(cx - half, 0)), int(min(cx + half, w))
    ya, yb = int(max(cy - half, 0)), int(min(cy + half, h))
    if xb - xa < 8 or yb - ya < 8:
        return None
    return cv2.resize(gray[ya:yb, xa:xb], (size, size), interpolation=cv2.INTER_AREA)
