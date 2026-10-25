"""Alarm: generated beep (no audio asset needed), local playback and browser HTML."""
import base64
import io
import shutil
import subprocess
import sys
import tempfile
import threading
import wave
from typing import Optional

import numpy as np

from .config import Config
from .detector import SLEEPING


def make_beep_wav(freq: int = 880, seconds: float = 1.2, sr: int = 22050) -> bytes:
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    gate = (np.sin(2 * np.pi * 3 * t) > 0).astype(float)          # 3 beeps per second
    tone = 0.6 * np.sin(2 * np.pi * freq * t) * gate
    pcm = (tone * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def beep_html() -> str:
    b64 = base64.b64encode(make_beep_wav()).decode()
    return f'<audio autoplay><source src="data:audio/wav;base64,{b64}" type="audio/wav"></audio>'


class Alarm:
    def __init__(self, cfg: Optional[Config] = None):
        self.cfg = cfg or Config()
        self._last: Optional[float] = None
        self._wav = make_beep_wav()

    def should_sound(self, state: str, t: float) -> bool:
        """True once per cooldown while the state is SLEEPING."""
        if state != SLEEPING:
            return False
        if self._last is None or t - self._last >= self.cfg.ALARM_COOLDOWN_S:
            self._last = t
            return True
        return False

    def play_local(self) -> None:
        threading.Thread(target=self._play, daemon=True).start()

    def _play(self) -> None:
        try:
            if sys.platform.startswith("win"):
                import winsound
                winsound.PlaySound(self._wav, winsound.SND_MEMORY)
                return
            player = shutil.which("afplay") or shutil.which("aplay") or shutil.which("paplay")
            if player:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    f.write(self._wav)
                subprocess.run([player, f.name], check=False, capture_output=True)
                return
        except Exception:
            pass
        print("\a", end="", flush=True)
