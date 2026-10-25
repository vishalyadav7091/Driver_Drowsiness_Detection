import io
import wave

from src.alarm import Alarm, make_beep_wav
from src.detector import ALERT, SLEEPING


def test_wav_is_valid():
    with wave.open(io.BytesIO(make_beep_wav()), "rb") as w:
        assert w.getnframes() > 0 and w.getframerate() == 22050


def test_alarm_cooldown():
    a = Alarm()
    assert a.should_sound(SLEEPING, 0.0)
    assert not a.should_sound(SLEEPING, 1.0)
    assert a.should_sound(SLEEPING, 2.5)
    assert not a.should_sound(ALERT, 10.0)
