from src.config import Config
from src.detector import ALERT, DROWSY, NO_FACE, SLEEPING, DrowsinessDetector

FPS = 30


def run(det, seconds, t0, ear, mar=0.1, pitch=0.0, cnn=None, face=True):
    res = None
    for i in range(int(seconds * FPS)):
        t = t0 + i / FPS
        res = det.update(t, face, ear if face else None, mar, pitch, cnn)
    return res, t0 + seconds


def test_open_eyes_stay_alert():
    det = DrowsinessDetector()
    res, _ = run(det, 5, 0, 0.30)
    assert res.state == ALERT and not res.alarm


def test_long_closure_triggers_sleeping_then_recovers():
    det = DrowsinessDetector()
    res, t = run(det, 2, 0, 0.30)
    res, t = run(det, 2.0, t, 0.10)
    assert res.state == SLEEPING and res.alarm
    res, t = run(det, 0.3, t, 0.30)
    assert res.state == SLEEPING            # hysteresis keeps the state briefly
    res, t = run(det, 2.5, t, 0.30)
    assert res.state in (ALERT, DROWSY)


def test_blink_is_counted_not_alarmed():
    det = DrowsinessDetector()
    res, t = run(det, 2, 0, 0.30)
    res, t = run(det, 0.2, t, 0.08)
    res, t = run(det, 1, t, 0.30)
    assert res.blinks == 1 and res.state == ALERT


def test_cnn_must_agree_with_ear():
    det = DrowsinessDetector()
    run(det, 1, 0, 0.30)
    res, _ = run(det, 3, 1, 0.10, cnn=0.05)   # EAR says closed, CNN says open
    assert res.state == ALERT


def test_cnn_confirms_borderline_ear():
    det = DrowsinessDetector()
    run(det, 1, 0, 0.30)
    res, _ = run(det, 2.5, 1, 0.24, cnn=0.95)  # EAR above threshold but borderline
    assert res.state == SLEEPING


def test_repeated_yawns_make_drowsy():
    det = DrowsinessDetector()
    t = 0.0
    for _ in range(2):
        _, t = run(det, 1.5, t, 0.30, mar=0.9)
        _, t = run(det, 1.0, t, 0.30, mar=0.1)
    res, _ = run(det, 0.5, t, 0.30)
    assert res.yawns == 2 and res.state == DROWSY


def test_no_face_state_after_timeout():
    det = DrowsinessDetector()
    run(det, 1, 0, 0.30)
    res, _ = run(det, 2, 1, 0, face=False)
    assert res.state == NO_FACE and not res.alarm


def test_calibration_sets_personal_threshold():
    det = DrowsinessDetector(Config(CAL_SECONDS=2))
    det.start_calibration()
    run(det, 3, 0, 0.40)
    assert abs(det.ear_thresh - 0.28) < 1e-9      # 0.72 * 0.40 = 0.288, clipped to 0.28
