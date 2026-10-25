"""Single source of truth for every threshold and path."""
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    # --- eye closure ---
    EAR_THRESH: float = 0.21        # below this the eye counts as closed
    CLOSURE_SECONDS: float = 1.5    # closure this long = SLEEPING
    SMOOTH_N: int = 5               # moving-average window (frames) for EAR
    BLINK_MAX_S: float = 0.7        # closures shorter than this are blinks
    # --- CNN fusion ---
    CNN_CONF: float = 0.85          # CNN alone may confirm closure above this ...
    CNN_BORDER: float = 1.25        # ... when EAR < threshold * this (borderline)
    CNN_EVERY: int = 2              # run the CNN every N frames
    EYE_SIZE: int = 64
    # --- yawn / head ---
    MAR_THRESH: float = 0.60
    YAWN_SECONDS: float = 1.0
    NOD_DEG: float = 20.0           # pitch deviation from baseline
    NOD_SECONDS: float = 1.0
    # --- drowsy rules ---
    PERCLOS_WINDOW_S: float = 45.0
    PERCLOS_MIN_S: float = 5.0      # ignore PERCLOS until this much history
    PERCLOS_DROWSY: float = 0.15
    EVENT_WINDOW_S: float = 60.0
    YAWNS_FOR_DROWSY: int = 2
    NODS_FOR_DROWSY: int = 2
    HOLD_SECONDS: float = 1.0       # hysteresis before dropping to a lower state
    NO_FACE_SECONDS: float = 1.0
    # --- calibration ---
    CAL_SECONDS: float = 5.0
    CAL_RATIO: float = 0.72         # personal threshold = ratio * median open EAR
    # --- alarm ---
    ALARM_COOLDOWN_S: float = 2.0
    # --- video ---
    RESIZE_WIDTH: int = 640
    # --- paths ---
    MODEL_PATH: Path = ROOT / "models" / "eye_state_cnn.keras"
    OUTPUT_DIR: Path = ROOT / "outputs"
