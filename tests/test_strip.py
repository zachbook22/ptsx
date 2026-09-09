"""Strip silence and clip-nudge tests."""

import numpy as np

from ptsx.layout import place_time_clips
from ptsx.strip import db_to_amp, exclusive_stripped, strip_silence
from ptsx.turns import TurnParams


def test_strip_threshold_and_pad():
    sr = 1000
    audio = np.zeros(sr, dtype=np.float32)
    # 200ms burst at -20 dBFS in the middle.
    amp = db_to_amp(-20.0)
    audio[400:600] = amp
    keep = strip_silence(audio, sr, thresh_db=-28.0, pad_s=0.250, hop_s=0.010)
    hop = 0.010
    on = np.where(keep)[0]
    assert on.size > 0
    start_s, end_s = on[0] * hop, (on[-1] + 1) * hop
    # Burst 0.40–0.60 plus 250ms each side → about 0.15–0.85
    assert start_s <= 0.16
    assert end_s >= 0.84
    # Below-threshold noise is not kept far away.
    assert not keep[0]
    assert not keep[-1]


def test_quiet_word_tail_is_held():
    sr = 1000
    audio = np.zeros(sr, dtype=np.float32)
    audio[400:550] = db_to_amp(-20.0)
    # Decaying last syllable below the -28 open threshold, above the hold.
    audio[550:700] = db_to_amp(-32.0)
    keep = strip_silence(audio, sr, thresh_db=-28.0, pad_s=0.250, hop_s=0.010)
    hop = 0.010
    on = np.where(keep)[0]
    end_s = (on[-1] + 1) * hop
    assert end_s >= 0.95


def test_below_threshold_is_stripped():
    sr = 1000
    audio = np.full(sr, db_to_amp(-40.0), dtype=np.float32)
    keep = strip_silence(audio, sr, thresh_db=-28.0, pad_s=0.250, hop_s=0.010)
    assert not keep.any()


def test_noisy_floor_does_not_hold_forever():
    from ptsx.strip import auto_hold_db, clips_from_mask

    sr = 1000
    audio = np.full(int(sr * 3.0), db_to_amp(-33.0), dtype=np.float32)
    audio[int(0.4 * sr) : int(0.7 * sr)] = db_to_amp(-12.0)
    audio[int(1.8 * sr) : int(2.1 * sr)] = db_to_amp(-12.0)
    hold = auto_hold_db(audio, sr, hop_s=0.010, thresh_db=-28.0)
    keep = strip_silence(
        audio, sr, thresh_db=-28.0, pad_s=0.050, hop_s=0.010, hold_db=hold
    )
    clips = clips_from_mask(keep, 0.010, "user")
    assert len(clips) >= 2
    assert all(end - start < 1.2 for _, start, end in clips)


def test_exclusive_overlap_picks_louder():
    hop = 0.010
    n = 100
    u = np.ones(n, dtype=bool)
    a = np.ones(n, dtype=bool)
    pu = np.full(n, 0.2, dtype=np.float32)
    pa = np.full(n, 0.05, dtype=np.float32)
    ku, ka = exclusive_stripped(u, a, pu, pa, hop, hang_s=0.05)
    assert ku.all()
    assert not ka.any()


def test_nested_sprawl_keeps_tighter_clip():
    hop = 0.010
    n = 700
    peak_u = np.zeros(n, dtype=np.float32)
    peak_a = np.zeros(n, dtype=np.float32)
    peak_u[10:220] = 0.08
    peak_a[5:610] = 0.10
    clips = [
        ("user", 0.10, 2.20),
        ("assistant", 0.05, 6.10),
    ]
    from ptsx.strip import resolve_nested_clips

    kept, dropped = resolve_nested_clips(clips, peak_u, peak_a, hop, max_backchannel_s=0.70)
    speakers = {c[0] for c in kept}
    assert "user" in speakers
    user_kept = [c for c in kept if c[0] == "user"]
    assert any(c[2] - c[1] > 1.5 for c in user_kept)
    # Sprawling assistant clip is punched around the user turn.
    asst_kept = [c for c in kept if c[0] == "assistant"]
    assert all(c[1] >= 2.15 - 1e-3 or c[2] <= 0.15 for c in asst_kept)


def test_nudge_does_not_shorten_source():
    clips = [("user", 0.16, 0.96), ("assistant", 0.99, 1.92)]
    placed = place_time_clips(clips, TurnParams(min_gap_s=0.3, target_gap_s=0.5))
    assert abs(placed[0].src_end_s - 0.96) < 1e-9
    assert abs(placed[1].src_start_s - 0.99) < 1e-9
    gap = placed[1].dst_start_s - placed[0].dst_end_s
    assert gap >= 0.3 - 1e-6


def test_nudge_packs_stripped_dead_air():
    clips = [("user", 0.0, 1.0), ("assistant", 7.0, 8.0)]
    placed = place_time_clips(
        clips, TurnParams(min_gap_s=0.3, max_gap_s=0.8, target_gap_s=0.5)
    )
    gap = placed[1].dst_start_s - placed[0].dst_end_s
    assert abs(gap - 0.5) < 1e-9


def test_merge_near_same_speaker():
    from ptsx.strip import merge_near_clips, resolve_nested_clips

    clips = [
        ("user", 0.08, 2.23),
        ("user", 2.37, 5.40),
        ("assistant", 0.06, 6.10),
    ]
    merged = merge_near_clips(clips, max_gap_s=0.30)
    user = [c for c in merged if c[0] == "user"]
    assert len(user) == 1
    assert abs(user[0][1] - 0.08) < 1e-9
    assert abs(user[0][2] - 5.40) < 1e-9

    hop = 0.010
    n = 700
    peak_u = np.zeros(n, dtype=np.float32)
    peak_a = np.zeros(n, dtype=np.float32)
    peak_u[8:540] = 0.08
    peak_a[6:610] = 0.10
    kept, _ = resolve_nested_clips(merged, peak_u, peak_a, hop, max_backchannel_s=0.70)
    asst = [c for c in kept if c[0] == "assistant"]
    # No 140ms sliver between the merged user turn.
    assert all(c[2] - c[1] >= 0.25 for c in asst)


def test_nudge_keeps_small_same_speaker_gap():
    clips = [("user", 0.0, 1.0), ("user", 1.14, 2.0)]
    placed = place_time_clips(
        clips,
        TurnParams(min_gap_s=0.3, max_gap_s=0.8, target_gap_s=0.5),
        dropped_s=[(1.0, 1.14)],
    )
    gap = placed[1].dst_start_s - placed[0].dst_end_s
    assert abs(gap - 0.14) < 1e-9
