"""Turn-assignment unit tests (no audio files required)."""

import numpy as np

from ptsx.render import exclusive_gains, keep_mask_to_gain
from ptsx.turns import TurnParams, assign_turns, chunk_sample_bounds
from ptsx.vad import mask_to_segments


def _mask(n, *spans):
    m = np.zeros(n, dtype=bool)
    for s, e in spans:
        m[s:e] = True
    return m


def test_backchannel_is_dropped_when_other_talks_through():
    hop = 0.032
    n = 200
    # User talks the whole time; assistant has a 0.32s burst in the middle.
    user = _mask(n, (10, 160))
    asst = _mask(n, (80, 90))  # 10 frames ≈ 0.32s
    result = assign_turns(user, asst, hop, TurnParams(max_backchannel_s=0.55))
    assert not result.keep_asst.any()
    assert result.keep_user[80:90].all()
    assert any(c.reason == "backchannel" for c in result.cuts)


def test_answered_interruption_is_kept_as_turn():
    hop = 0.032
    n = 200
    user = _mask(n, (10, 70), (120, 170))  # stops, then resumes after assistant
    asst = _mask(n, (80, 110))
    result = assign_turns(user, asst, hop)
    assert result.keep_asst[85:100].all()
    assert not np.any(result.keep_user & result.keep_asst)


def test_no_overlap_and_outgoing_tail_kept():
    hop = 0.032
    n = 80
    user = _mask(n, (5, 30))
    asst = _mask(n, (31, 60))  # 1 frame gap ≈ 32ms
    result = assign_turns(
        user,
        asst,
        hop,
        TurnParams(min_gap_s=0.3, target_gap_s=0.5, max_gap_s=0.8),
    )
    assert not np.any(result.keep_user & result.keep_asst)
    u_end = mask_to_segments(result.keep_user)[-1][1]
    # Tail hang may extend past 30, but must not be trimmed back.
    assert u_end >= 30
    a_start = mask_to_segments(result.keep_asst)[0][0]
    assert a_start >= u_end
    gap = (a_start - u_end) * hop
    assert gap >= 0.3 - hop


def test_original_timeline_identity_clips():
    hop = 0.032
    segs = [("user", 5, 30), ("assistant", 31, 60)]
    from ptsx.layout import place_segments
    from ptsx.turns import TurnParams

    clips = place_segments(segs, hop, TurnParams(min_gap_s=0.3, target_gap_s=0.5))
    assert abs(clips[0].src_end_s - 30 * hop) < 1e-9
    assert abs(clips[1].src_start_s - 31 * hop) < 1e-9
    assert clips[1].dst_start_s >= clips[0].dst_end_s


def test_overlap_does_not_chatter():
    hop = 0.032
    n = 80
    user = np.ones(n, dtype=bool)
    asst = np.ones(n, dtype=bool)
    ru = np.full(n, 0.05, dtype=np.float32)
    ra = np.full(n, 0.01, dtype=np.float32)
    result = assign_turns(user, asst, hop, user_rms=ru, asst_rms=ra)
    assert result.keep_user.all()
    assert not result.keep_asst.any()
    asst_cuts = [c for c in result.cuts if c.speaker == "assistant"]
    assert len(asst_cuts) <= 2


def test_gains_never_overlap():
    hop = 0.032
    sr = 48000
    n = int(2.0 * sr)
    mask_u = _mask(40, (5, 20))
    mask_a = _mask(40, (22, 35))
    gu = keep_mask_to_gain(mask_u, hop, sr, n, fade_ms=10)
    ga = keep_mask_to_gain(mask_a, hop, sr, n, fade_ms=10)
    gu, ga = exclusive_gains(gu, ga)
    assert not np.any((gu > 0) & (ga > 0))


def test_gate_nudges_without_overlap():
    sr = 8000
    n = sr * 2
    user = np.zeros(n, dtype=np.float32)
    asst = np.zeros(n, dtype=np.float32)
    t = np.arange(n) / sr
    user[(t >= 0.1) & (t < 0.6)] = 0.2 * np.sin(2 * np.pi * 220 * t[(t >= 0.1) & (t < 0.6)])
    asst[(t >= 0.9) & (t < 1.4)] = 0.2 * np.sin(2 * np.pi * 140 * t[(t >= 0.9) & (t < 1.4)])
    from ptsx.pipeline import gate_arrays

    u_out, a_out, comb, u_rem, a_rem, cuts, clips = gate_arrays(user, asst, sr)
    assert len(u_rem) == n
    assert len(a_rem) == n
    assert len(u_out) == len(a_out) == len(comb)
    assert not np.any((np.abs(u_out) > 0) & (np.abs(a_out) > 0))
    assert clips
    assert clips[0].src_end_s - clips[0].src_start_s > 0.4


def test_chunk_bounds_snap_to_turn_gap():
    hop = 0.032
    sr = 1000
    n = 100_000  # 100 seconds at 1 kHz
    # User 0-40s, assistant 41-90s at 32ms frames
    n_frames = int(n / (hop * sr))
    user = _mask(n_frames, (0, int(40 / hop)))
    asst = _mask(n_frames, (int(41 / hop), int(90 / hop)))
    bounds = chunk_sample_bounds(user, asst, hop, sr, n, chunk_minutes=0.5)
    assert bounds[0][0] == 0
    assert bounds[-1][1] == n
    # A 30s chunk should not land at exactly 30s if 40s is the turn gap.
    if len(bounds) > 1:
        assert bounds[0][1] > 30 * sr
