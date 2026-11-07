"""Synthetic test recordings for the upload + apnea/AHI test (not real patients).

tests/audio/base_test.wav : 2 min of regular breathing, 15 breaths/min (4 s cycle)
tests/audio/long_test.wav : 12 min of the same breathing with 11 silent 15-s pauses (one every minute from 1:00)
                            -> the AASM scorer should find exactly 11 apneas.
8 kHz, 16-bit mono.
"""
import wave
from pathlib import Path
import numpy as np
from scipy.signal import butter, sosfilt

SR = 8000
OUT = Path(__file__).resolve().parent / 'audio'


def breathing(seconds, pauses=(), seed=0):
    rng = np.random.default_rng(seed)
    n = int(seconds * SR); t = np.arange(n) / SR
    noise = sosfilt(butter(4, [150, 1200], 'bandpass', fs=SR, output='sos'), rng.standard_normal(n))
    ph = t % 4.0                                                   # 4-s cycle: 1.6 s inspiration, 2.0 s expiration, 0.4 s rest
    env = np.where(ph < 1.6, np.sin(np.pi * ph / 1.6), np.where(ph < 3.6, 0.6 * np.sin(np.pi * (ph - 1.6) / 2.0), 0.0)) ** 1.5
    for start, dur in pauses:
        env[(t >= start) & (t < start + dur)] = 0.0
    x = noise * env + 0.004 * rng.standard_normal(n)               # quiet sensor floor
    return x / np.abs(x).max() * 0.5


def write(path, x):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), 'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes((x * 32767).astype('<i2').tobytes())


if __name__ == '__main__':
    write(OUT / 'base_test.wav', breathing(120, seed=1))
    write(OUT / 'long_test.wav', breathing(720, pauses=[(60 * k - 0.6, 15.6) for k in range(1, 12)], seed=2))
    print('wrote', OUT)
