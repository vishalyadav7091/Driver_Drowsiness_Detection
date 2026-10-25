"""Fusion of EAR, CNN, yawns and head nods into a driver state."""
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional

import numpy as np

from .config import Config

ALERT, DROWSY, SLEEPING = "ALERT", "DROWSY", "SLEEPING"
NO_FACE, CALIBRATING = "NO FACE", "CALIBRATING"
_RANK = {ALERT: 0, DROWSY: 1, SLEEPING: 2}


@dataclass
class Result:
    t: float
    state: str
    face: bool
    ear: float
    ear_smooth: float
    mar: float
    pitch: float
    cnn_closed: float
    eye_closed: bool
    closure_s: float
    perclos: float
    blinks: int
    yawns: int
    nods: int
    alarm: bool


class DrowsinessDetector:
    def __init__(self, cfg: Optional[Config] = None):
        self.cfg = cfg or Config()
        self.ear_thresh = self.cfg.EAR_THRESH
        self.reset()

    # ------------------------------------------------------------------
    def reset(self) -> None:
        c = self.cfg
        self._ears: Deque[float] = deque(maxlen=c.SMOOTH_N)
        self._samples: Deque[tuple] = deque()
        self._closed_since: Optional[float] = None
        self._yawn_since: Optional[float] = None
        self._yawn_on = False
        self._nod_since: Optional[float] = None
        self._nod_on = False
        self._yawn_events: Deque[float] = deque()
        self._nod_events: Deque[float] = deque()
        self._no_face_since: Optional[float] = None
        self._lower_since: Optional[float] = None
        self._cal: Optional[dict] = None
        self.pitch_base = 0.0
        self.blinks = self.yawns = self.nods = 0
        self.state = ALERT

    def start_calibration(self) -> None:
        """Look straight ahead with eyes open for CAL_SECONDS."""
        self._cal = {"start": None, "ears": [], "pitches": []}

    # ------------------------------------------------------------------
    def _result(self, t, state, face=False, ear=0.0, ear_s=0.0, mar=0.0, pitch=0.0,
                cnn=-1.0, closed=False, closure=0.0, perclos=0.0) -> Result:
        return Result(t, state, face, ear, ear_s, mar, pitch, cnn, closed, closure, perclos,
                      self.blinks, self.yawns, self.nods, state == SLEEPING)

    @staticmethod
    def _timed_flag(cond, t, since, on, hold):
        """Return (since, on, fired) for a condition that must persist `hold` s."""
        if not cond:
            return None, False, False
        since = t if since is None else since
        if not on and t - since >= hold:
            return since, True, True
        return since, on, False

    # ------------------------------------------------------------------
    def update(self, t: float, face_found: bool, ear: Optional[float] = None,
               mar: Optional[float] = None, pitch: Optional[float] = None,
               cnn_closed: Optional[float] = None) -> Result:
        c = self.cfg

        # ---- no face ----
        if not face_found or ear is None:
            self._closed_since = None
            self._no_face_since = t if self._no_face_since is None else self._no_face_since
            if t - self._no_face_since >= c.NO_FACE_SECONDS:
                return self._result(t, NO_FACE)
            return self._result(t, self.state)
        self._no_face_since = None
        mar = mar or 0.0
        pitch = pitch or 0.0

        # ---- calibration ----
        if self._cal is not None:
            if self._cal["start"] is None:
                self._cal["start"] = t
            self._cal["ears"].append(ear)
            self._cal["pitches"].append(pitch)
            if t - self._cal["start"] >= c.CAL_SECONDS:
                med = float(np.median(self._cal["ears"]))
                self.ear_thresh = float(np.clip(c.CAL_RATIO * med, 0.15, 0.28))
                self.pitch_base = float(np.median(self._cal["pitches"]))
                self._cal = None
            return self._result(t, CALIBRATING, True, ear, ear, mar, pitch)

        # ---- eye closed? (EAR fused with CNN) ----
        self._ears.append(ear)
        ear_s = float(np.mean(self._ears))
        ear_closed = ear_s < self.ear_thresh
        if cnn_closed is None:
            closed = ear_closed
        elif ear_closed:
            closed = cnn_closed >= 0.5                       # both must agree
        else:
            closed = (cnn_closed >= c.CNN_CONF) and ear_s < self.ear_thresh * c.CNN_BORDER

        # ---- closure timer / blinks ----
        closure_s = 0.0
        if closed:
            self._closed_since = t if self._closed_since is None else self._closed_since
            closure_s = t - self._closed_since
        elif self._closed_since is not None:
            if 0.05 <= t - self._closed_since < c.BLINK_MAX_S:
                self.blinks += 1
            self._closed_since = None

        # ---- PERCLOS ----
        self._samples.append((t, closed))
        while self._samples and t - self._samples[0][0] > c.PERCLOS_WINDOW_S:
            self._samples.popleft()
        span = t - self._samples[0][0]
        perclos = float(np.mean([s[1] for s in self._samples]))
        perclos_ok = span >= c.PERCLOS_MIN_S

        # ---- yawns and nods ----
        self._yawn_since, self._yawn_on, fired = self._timed_flag(
            mar > c.MAR_THRESH, t, self._yawn_since, self._yawn_on, c.YAWN_SECONDS)
        if fired:
            self.yawns += 1
            self._yawn_events.append(t)
        self._nod_since, self._nod_on, fired = self._timed_flag(
            abs(pitch - self.pitch_base) > c.NOD_DEG, t, self._nod_since, self._nod_on, c.NOD_SECONDS)
        if fired:
            self.nods += 1
            self._nod_events.append(t)
        for q in (self._yawn_events, self._nod_events):
            while q and t - q[0] > c.EVENT_WINDOW_S:
                q.popleft()

        # ---- raw state, then hysteresis ----
        if closure_s >= c.CLOSURE_SECONDS:
            raw = SLEEPING
        elif ((perclos_ok and perclos > c.PERCLOS_DROWSY)
              or len(self._yawn_events) >= c.YAWNS_FOR_DROWSY
              or len(self._nod_events) >= c.NODS_FOR_DROWSY):
            raw = DROWSY
        else:
            raw = ALERT

        cur = self.state if self.state in _RANK else ALERT
        if _RANK[raw] >= _RANK[cur]:
            self.state, self._lower_since = raw, None
        else:
            self._lower_since = t if self._lower_since is None else self._lower_since
            if t - self._lower_since >= c.HOLD_SECONDS:
                self.state, self._lower_since = raw, None

        return self._result(t, self.state, True, ear, ear_s, mar, pitch,
                            -1.0 if cnn_closed is None else cnn_closed,
                            closed, closure_s, perclos if perclos_ok else 0.0)
