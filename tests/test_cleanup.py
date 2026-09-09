import numpy as np
import pytest

from ptsx.cleanup import apply_loudness, true_peak_db


def _tone(sr: int, seconds: float, db: float) -> np.ndarray:
    amp = 10.0 ** (db / 20.0)
    t = np.arange(int(sr * seconds), dtype=np.float64) / sr
    return (amp * np.sin(2 * np.pi * 1000.0 * t)).astype(np.float32)


def test_loudness_caps_true_peak():
    sr = 48000
    x = _tone(sr, 1.5, -1.0)
    y = apply_loudness(x, sr, target_lufs=-21.9, true_peak_db_lim=-3.1)
    assert true_peak_db(y, sr) <= -3.05


def test_loudness_raises_quiet_speech():
    sr = 48000
    x = _tone(sr, 1.5, -40.0)
    y = apply_loudness(x, sr, target_lufs=-21.9, true_peak_db_lim=-3.1)
    assert float(np.max(np.abs(y))) > float(np.max(np.abs(x)))
    assert true_peak_db(y, sr) <= -3.05


def test_silence_stays_silence():
    sr = 48000
    x = np.zeros(sr, dtype=np.float32)
    y = apply_loudness(x, sr)
    assert float(np.max(np.abs(y))) < 1e-6


def test_plugin_paths_exist_or_skip():
    from ptsx.cleanup import CLEANUP_PLUGIN_PATHS, cleanup_plugins_present

    if not cleanup_plugins_present():
        pytest.skip("cleanup VST3s are not installed")
    assert all(p.exists() for p in CLEANUP_PLUGIN_PATHS)
