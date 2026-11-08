"""Synthetic test recordings for the upload + apnea/AHI test (not real patients).

tests/audio/base_test.wav : 2 min of regular breathing, 15 breaths/min (4 s cycle)
tests/audio/long_test.wav : 12 min of the same breathing with 11 silent 15-s pauses (one every minute from 1:00)
                            -> the AASM scorer should find exactly 11 apneas.
tests/audio/quiet_test.wav: 12 min of regular breathing with NO pauses -> the scorer must find 0 events
                            (specificity: nothing else in the suite catches a false-positive regression).
tests/audio/crk_<idw>.wav : damped sinusoids whose first half-cycle is exactly <idw> ms, for the CORSA
                            IDW calibration in tests/corsa_calib.js.
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


def crackles(idw_ms, reps=40, seconds=12, seed=0):
    """Damped sinusoid whose FIRST HALF-CYCLE is exactly idw_ms — a crackle of known IDW by construction."""
    rng = np.random.default_rng(seed); n = int(seconds * SR)
    x = 0.002 * rng.standard_normal(n)
    freq = 1000.0 / (2 * idw_ms)                       # half-cycle of width idw -> sine at 1/(2*idw)
    L = int(round(0.012 * SR)); t = np.arange(L) / SR
    w = np.sin(2 * np.pi * freq * t) * np.exp(-t / 0.0025)
    for k in range(reps):
        a = int((0.2 + k * 0.28) * SR)
        if a + L < n: x[a:a + L] += 0.6 * w
    return x / np.abs(x).max() * 0.7


IDWS = (0.50, 0.70, 1.00, 1.25, 2.00)

if __name__ == '__main__':
    write(OUT / 'base_test.wav', breathing(120, seed=1))
    write(OUT / 'long_test.wav', breathing(720, pauses=[(60 * k - 0.6, 15.6) for k in range(1, 12)], seed=2))
    write(OUT / 'quiet_test.wav', breathing(720, seed=3))
    for idw in IDWS: write(OUT / f'crk_{idw:.2f}.wav', crackles(idw))
    print('wrote', OUT)
