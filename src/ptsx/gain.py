"""Keep speech peaks in a listen window: boost below -6 dBFS, limit above -3 dBFS."""

from __future__ import annotations

import numpy as np

from ptsx.strip import frame_peak

DEFAULT_MIN_DB = -6.0
DEFAULT_MAX_DB = -3.0
MAX_GAIN_DB = 36.0


def db_to_amp(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def amp_to_db(amp: float | np.ndarray) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(np.asarray(amp, dtype=np.float64), 1e-12))


def gain_db_for_peak(peak_db: np.ndarray, min_db: float, max_db: float) -> np.ndarray:
    """Per-frame gain so peaks land in [min_db, max_db]. Unchanged in between."""
    out = np.zeros(peak_db.shape, dtype=np.float64)
    out[peak_db < min_db] = min_db - peak_db[peak_db < min_db]
    out[peak_db > max_db] = max_db - peak_db[peak_db > max_db]
    return out


def auto_speech_db(peak_db: np.ndarray, strip_db: float = -28.0) -> float:
    """Open AGC on frames that look like this track's speech, not its noise floor."""
    p85 = float(np.percentile(peak_db, 85))
    if p85 < -55.0:
        return float(strip_db)
    return float(min(strip_db, p85))


def _hold_gain_db(gain_db: np.ndarray, speech: np.ndarray, hold: int) -> np.ndarray:
    """Keep the last speech gain through short dips so words don't pump."""
    out = np.zeros_like(gain_db)
    last = 0.0
    below = hold + 1
    for i, on in enumerate(speech):
        if on:
            last = gain_db[i]
            below = 0
            out[i] = last
            continue
        below += 1
        out[i] = last if below <= hold else 0.0
    return out


def conform_level(
    audio: np.ndarray,
    sr: int,
    min_db: float = DEFAULT_MIN_DB,
    max_db: float = DEFAULT_MAX_DB,
    hop_s: float = 0.010,
    smooth_s: float = 0.080,
    hold_s: float = 0.150,
    max_gain_db: float = MAX_GAIN_DB,
    speech_db: float | None = None,
    strip_db: float = -28.0,
) -> np.ndarray:
    """Windowed peak conform. Silence is not boosted; remaining peaks cap at max_db."""
    if audio.size == 0:
        return audio
    peak = frame_peak(audio, sr, hop_s)
    if peak.size == 0:
        return audio.astype(np.float32, copy=False)
    peak_db = amp_to_db(peak)
    if speech_db is None:
        speech_db = auto_speech_db(peak_db, strip_db)
    speech = peak_db >= speech_db
    gain_db = gain_db_for_peak(peak_db, min_db, max_db)
    gain_db = np.where(speech, gain_db, 0.0)
    gain_db = np.clip(gain_db, -max_gain_db, max_gain_db)
    hold = max(1, int(round(hold_s / hop_s)))
    gain_db = _hold_gain_db(gain_db, speech, hold)
    win = max(1, int(round(smooth_s / hop_s)))
    kernel = np.ones(win, dtype=np.float64) / win
    gain_db = np.convolve(gain_db, kernel, mode="same")
    hop = max(1, int(round(sr * hop_s)))
    lin = np.repeat(db_to_amp_arr(gain_db), hop)
    if len(lin) < len(audio):
        lin = np.pad(lin, (0, len(audio) - len(lin)), constant_values=1.0)
    else:
        lin = lin[: len(audio)]
    out = audio.astype(np.float64) * lin
    cap = db_to_amp(max_db)
    np.clip(out, -cap, cap, out=out)
    return np.ascontiguousarray(out, dtype=np.float32)


def db_to_amp_arr(db: np.ndarray) -> np.ndarray:
    return np.power(10.0, db / 20.0)


def needs_detection_boost(
    audio: np.ndarray,
    sr: int,
    hop_s: float = 0.010,
    strip_db: float = -28.0,
    min_speech_frac: float = 0.05,
) -> bool:
    """True when almost nothing on this track crosses the strip threshold."""
    peak = frame_peak(audio, sr, hop_s)
    if peak.size == 0:
        return False
    frac = float(np.mean(peak >= db_to_amp(strip_db)))
    return frac < min_speech_frac
