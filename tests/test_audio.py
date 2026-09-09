"""WAV round-trip and dual-track alignment."""

from pathlib import Path

import numpy as np

from ptsx.align import align_exclusive
from ptsx.audio import load_mono, write_wav_24
from ptsx.strip import db_to_amp, frame_peak


def test_pcm24_roundtrip(tmp_path: Path):
    sr = 48000
    t = np.arange(sr, dtype=np.float32) / sr
    x = (0.25 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    path = tmp_path / "tone.wav"
    write_wav_24(path, x, sr)
    y, sr2 = load_mono(path)
    assert sr2 == sr
    assert abs(len(y) - len(x)) <= 1
    n = min(len(x), len(y))
    assert float(np.max(np.abs(x[:n] - y[:n]))) < 2 / 8388608 * 4


def test_align_pads_late_user():
    sr = 1000
    user = np.zeros(sr * 4, dtype=np.float32)
    asst = np.zeros(sr * 4, dtype=np.float32)
    # Same 0.5 s burst looks simultaneous in the files; user actually started 1 s late.
    user[int(1.0 * sr) : int(1.5 * sr)] = 0.2
    asst[int(1.0 * sr) : int(1.5 * sr)] = 0.2
    u2, a2, lag = align_exclusive(user, asst, sr, hop_s=0.010, max_lag_s=2.0)
    assert abs(len(u2) - len(a2)) == 0
    hop_s = 0.010
    u_on = frame_peak(u2, sr, hop_s) >= db_to_amp(-28)
    a_on = frame_peak(a2, sr, hop_s) >= db_to_amp(-28)
    n = min(len(u_on), len(a_on))
    overlap = float(np.mean(u_on[:n] & a_on[:n]))
    assert overlap < 0.05
