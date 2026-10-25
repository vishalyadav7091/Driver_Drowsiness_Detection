
"""Alarm utilities for the Driver Drowsiness Detection project."""

import base64
import io
import shutil
import subprocess
import sys
import tempfile
import threading
import wave
from pathlib import Path
from typing import Optional

import numpy as np

from .config import Config
from .detector import SLEEPING


def make_beep_wav(
    freq: int = 1000,
    seconds: float = 5.0,
    sr: int = 22050,
) -> bytes:
    """Generate a repeating warning beep as WAV audio."""
    if freq <= 0 or seconds <= 0 or sr <= 0:
        raise ValueError("freq, seconds, and sr must be positive.")

    t = np.arange(int(sr * seconds), dtype=np.float32) / sr

    # Three short beeps per second.
    beep_pattern = (np.sin(2 * np.pi * 3 * t) > 0).astype(np.float32)
    tone = 0.6 * np.sin(2 * np.pi * freq * t) * beep_pattern

    # Fade edges to reduce clicking sounds.
    fade_samples = min(int(sr * 0.01), len(tone) // 2)
    if fade_samples:
        fade = np.linspace(0, 1, fade_samples, dtype=np.float32)
        tone[:fade_samples] *= fade
        tone[-fade_samples:] *= fade[::-1]

    pcm = (tone * 32767).astype(np.int16)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sr)
        wav.writeframes(pcm.tobytes())

    return buffer.getvalue()


def beep_html(seconds: float = 5.0, freq: int = 1000) -> str:
    """Return browser-playable audio HTML for Streamlit."""
    audio_b64 = base64.b64encode(
        make_beep_wav(freq=freq, seconds=seconds)
    ).decode("ascii")

    return (
        '<audio autoplay controls>'
        f'<source src="data:audio/wav;base64,{audio_b64}" '
        'type="audio/wav">'
        'Your browser does not support audio.'
        '</audio>'
    )


class Alarm:
    """Play an alarm when the detector reports a sleeping state."""

    def __init__(
        self,
        cfg: Optional[Config] = None,
        freq: int = 1000,
        duration: float = 5.0,
    ):
        self.cfg = cfg or Config()
        self._last: Optional[float] = None
        self._wav = make_beep_wav(freq=freq, seconds=duration)
        self._lock = threading.Lock()

    def should_sound(self, state: str, t: float) -> bool:
        """Trigger once per cooldown while the state is SLEEPING."""
        if state != SLEEPING:
            return False

        with self._lock:
            if (
                self._last is None
                or t - self._last >= self.cfg.ALARM_COOLDOWN_S
            ):
                self._last = t
                return True

        return False

    def play_local(self) -> None:
        """Play audio on the computer running this Python process."""
        threading.Thread(target=self._play, daemon=True).start()

    def _play(self) -> None:
        try:
            if sys.platform.startswith("win"):
                import winsound

                winsound.PlaySound(
                    self._wav,
                    winsound.SND_MEMORY | winsound.SND_NODEFAULT,
                )
                return

            player = (
                shutil.which("afplay")
                or shutil.which("aplay")
                or shutil.which("paplay")
            )

            if player:
                with tempfile.NamedTemporaryFile(
                    suffix=".wav", delete=False
                ) as audio_file:
                    audio_file.write(self._wav)
                    audio_path = Path(audio_file.name)

                try:
                    subprocess.run(
                        [player, str(audio_path)],
                        check=False,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                finally:
                    audio_path.unlink(missing_ok=True)
                return

        except Exception as exc:
            print(f"Could not play alarm: {exc}")

        # Fallback terminal bell; it may not produce an audible sound.
        print("\a", end="", flush=True)
