"""Align two speaker recordings that were not started together."""

from __future__ import annotations

import numpy as np

from ptsx.audio import match_lengths
from ptsx.strip import db_to_amp, frame_peak


def align_exclusive(
    user: np.ndarray,
    asst: np.ndarray,
    sr: int,
    hop_s: float = 0.010,
    thresh_db: float = -28.0,
    max_lag_s: float = 120.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Pad one start so the two tracks are complementary (one speaker at a time).

    Negative lag: pad the user (user hit record later). Positive: pad assistant.
    """
    u = (frame_peak(user, sr, hop_s) >= db_to_amp(thresh_db)).astype(np.float32)
    a = (frame_peak(asst, sr, hop_s) >= db_to_amp(thresh_db)).astype(np.float32)
    if u.size == 0 or a.size == 0:
        user, asst = match_lengths(user, asst)
        return user, asst, 0.0
    score = np.correlate(u, 1.0 - a, mode="full") + np.correlate(1.0 - u, a, mode="full")
    center = len(a) - 1
    lags = np.arange(len(score)) - center
    max_h = max(1, int(round(max_lag_s / hop_s)))
    win = (lags >= -max_h) & (lags <= max_h)
    lag_h = int(lags[win][int(np.argmax(score[win]))])
    lag_s = lag_h * hop_s
    pad = int(round(abs(lag_s) * sr))
    if pad:
        z = np.zeros(pad, dtype=np.float32)
        if lag_h < 0:
            user = np.concatenate([z, user.astype(np.float32, copy=False)])
        else:
            asst = np.concatenate([z, asst.astype(np.float32, copy=False)])
    user, asst = match_lengths(user, asst)
    return (
        np.ascontiguousarray(user, dtype=np.float32),
        np.ascontiguousarray(asst, dtype=np.float32),
        float(lag_s),
    )
