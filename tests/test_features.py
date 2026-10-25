import numpy as np

from src.features import eye_aspect_ratio, mouth_aspect_ratio


def eye(open_h: float) -> np.ndarray:
    # p1 outer, p2/p3 upper, p4 inner, p5/p6 lower; width 10
    return np.array([(0, 0), (3, -open_h), (7, -open_h), (10, 0), (7, open_h), (3, open_h)], float)


def test_ear_open_is_larger_than_closed():
    assert eye_aspect_ratio(eye(3.0)) > 0.5 > eye_aspect_ratio(eye(0.5)) > 0


def test_ear_known_value():
    # vertical gaps 2*1.5 each -> (3+3)/(2*10) = 0.3
    assert abs(eye_aspect_ratio(eye(1.5)) - 0.3) < 1e-9


def test_ear_degenerate_width_returns_zero():
    assert eye_aspect_ratio(np.zeros((6, 2))) == 0.0


def test_mar_grows_with_mouth_opening():
    def mouth(gap):
        lm = np.zeros((500, 2))
        lm[78], lm[308] = (0, 0), (10, 0)
        for (a, b), x in zip(((81, 178), (13, 14), (311, 402)), (2, 5, 8)):
            lm[a], lm[b] = (x, -gap / 2), (x, gap / 2)
        return lm
    assert mouth_aspect_ratio(mouth(8)) > mouth_aspect_ratio(mouth(1))
    assert abs(mouth_aspect_ratio(mouth(6)) - 0.6) < 1e-9


def test_head_pose_recovers_pitch_and_yaw():
    import pytest
    cv2 = pytest.importorskip("cv2")
    from src.features import _MODEL_3D, _POSE_IDX, head_pose

    w, h = 640, 480
    cam = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1.0]])
    flip = np.diag([1, -1, -1.0])           # model frame -> camera frame

    def landmarks(pitch, yaw):
        rx = cv2.Rodrigues(np.array([np.radians(pitch), 0, 0.0]))[0]
        ry = cv2.Rodrigues(np.array([0, np.radians(yaw), 0.0]))[0]
        pts = ((ry @ rx) @ (_MODEL_3D @ flip.T).T).T + np.array([0, 0, 600.0])
        uv = (cam @ pts.T).T
        uv = uv[:, :2] / uv[:, 2:3]
        lm = np.zeros((478, 2))
        for i, q in zip(_POSE_IDX, uv):
            lm[i] = q
        return lm

    pitch, yaw, _ = head_pose(landmarks(20, 0), w, h)
    assert abs(pitch - 20) < 1
    pitch, yaw, _ = head_pose(landmarks(0, 25), w, h)
    assert abs(yaw - 25) < 1 and abs(pitch) < 1
