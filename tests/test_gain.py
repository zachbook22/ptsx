import numpy as np

from ptsx.gain import conform_level, gain_db_for_peak


def _tone(sr: int, seconds: float, db: float) -> np.ndarray:
    amp = 10.0 ** (db / 20.0)
    t = np.arange(int(sr * seconds), dtype=np.float32) / sr
    return (amp * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)


def _peak_db(x: np.ndarray) -> float:
    return float(20.0 * np.log10(float(np.max(np.abs(x))) + 1e-12))


def test_gain_db_for_peak_window():
    peaks = np.array([-30.0, -4.5, -1.0])
    g = gain_db_for_peak(peaks, -6.0, -3.0)
    assert abs(g[0] - 24.0) < 1e-6
    assert abs(g[1]) < 1e-6
    assert abs(g[2] - (-2.0)) < 1e-6


def test_quiet_speech_boosts_to_minus_6():
    sr = 48000
    x = _tone(sr, 0.6, -30.0)
    y = conform_level(x, sr, speech_db=-40.0)
    assert -7.0 <= _peak_db(y) <= -2.5


def test_hot_speech_limits_to_minus_3():
    sr = 48000
    x = _tone(sr, 0.6, -0.5)
    y = conform_level(x, sr, speech_db=-40.0)
    assert _peak_db(y) <= -2.9


def test_in_window_unchanged():
    sr = 48000
    x = _tone(sr, 0.6, -4.5)
    y = conform_level(x, sr, speech_db=-40.0)
    assert abs(_peak_db(y) - _peak_db(x)) < 0.4


def test_silence_not_boosted():
    sr = 48000
    x = _tone(sr, 0.6, -70.0)
    y = conform_level(x, sr, speech_db=-28.0)
    assert _peak_db(y) < -60.0


def test_quiet_track_needs_detection_boost():
    from ptsx.gain import needs_detection_boost

    sr = 48000
    quiet = _tone(sr, 1.0, -36.0)
    loud = _tone(sr, 1.0, -12.0)
    assert needs_detection_boost(quiet, sr) is True
    assert needs_detection_boost(loud, sr) is False
