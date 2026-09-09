"""Script-first turn assignment: keep full unique turns, drop nested yeahs."""

import numpy as np

from ptsx.script import assign_script_turns, uniqueness_db


def test_opening_unique_turns_are_kept_full():
    hop = 0.010
    n = 1200
    peak_u = np.zeros(n, dtype=np.float32)
    peak_a = np.zeros(n, dtype=np.float32)
    peak_u[8:540] = 0.20
    peak_a[564:1169] = 0.20
    clips = [
        ("user", 0.08, 5.40),
        ("assistant", 5.64, 11.69),
    ]
    kept, dropped = assign_script_turns(clips, peak_u, peak_a, hop)
    assert kept == [("user", 0.08, 5.40), ("assistant", 5.64, 11.69)]
    assert dropped == []


def test_nested_yeah_is_dropped_floor_not_punched():
    hop = 0.010
    n = 2500
    peak_u = np.zeros(n, dtype=np.float32)
    peak_a = np.zeros(n, dtype=np.float32)
    peak_u[1231:2408] = 0.20
    peak_a[2136:2226] = 0.05  # quieter yeah on the other track
    clips = [
        ("user", 12.31, 24.08),
        ("assistant", 21.36, 22.26),
    ]
    kept, dropped = assign_script_turns(clips, peak_u, peak_a, hop)
    assert kept == [("user", 12.31, 24.08)]
    assert any(c[0] == "assistant" for c in dropped)


def test_bleed_clip_on_wrong_track_is_dropped():
    hop = 0.010
    n = 700
    peak_u = np.zeros(n, dtype=np.float32)
    peak_a = np.zeros(n, dtype=np.float32)
    peak_u[10:610] = 0.04  # bleed
    peak_a[10:610] = 0.20  # real assistant
    clips = [("user", 0.10, 6.10), ("assistant", 0.10, 6.10)]
    kept, dropped = assign_script_turns(clips, peak_u, peak_a, hop)
    assert kept == [("assistant", 0.10, 6.10)]
    assert any(c[0] == "user" for c in dropped)


def test_glue_backchannel_dropped_answered_yeah_kept():
    hop = 0.010
    n = 3000
    peak_u = np.zeros(n, dtype=np.float32)
    peak_a = np.zeros(n, dtype=np.float32)
    peak_u[100:2400] = 0.20
    peak_a[2410:2490] = 0.15
    peak_u[2635:2800] = 0.20
    clips = [
        ("user", 1.00, 24.00),
        ("assistant", 24.10, 24.90),
        ("user", 26.35, 28.00),
    ]
    kept, dropped = assign_script_turns(clips, peak_u, peak_a, hop)
    assert not any(c[0] == "assistant" and c[2] - c[1] < 1.0 for c in kept)
    assert any(c[0] == "assistant" for c in dropped)

    peak_u[:] = 0
    peak_a[:] = 0
    peak_a[1465:1501] = 0.20
    peak_u[1568:1634] = 0.20
    peak_a[1640:1750] = 0.20
    clips = [
        ("assistant", 14.65, 15.01),
        ("user", 15.68, 16.34),
        ("assistant", 16.40, 17.50),
    ]
    kept, dropped = assign_script_turns(clips, peak_u, peak_a, hop)
    assert any(c[0] == "user" for c in kept)


def test_pad_overlap_does_not_drop_next_turn():
    hop = 0.010
    n = 1200
    peak_u = np.zeros(n, dtype=np.float32)
    peak_a = np.zeros(n, dtype=np.float32)
    peak_u[8:540] = 0.20
    peak_a[564:1172] = 0.18
    clips = [
        ("user", 0.08, 5.68),
        ("assistant", 5.64, 11.72),
    ]
    kept, dropped = assign_script_turns(clips, peak_u, peak_a, hop)
    assert kept[0] == ("user", 0.08, 5.68)
    assert kept[1][0] == "assistant"
    assert kept[1][1] <= 5.64 + 1e-6
    assert kept[1][2] == 11.72
    assert not any(c[0] == "assistant" for c in dropped)


def test_uniqueness_digital_silence_is_high():
    hop = 0.010
    n = 600
    peak_u = np.zeros(n, dtype=np.float32)
    peak_a = np.zeros(n, dtype=np.float32)
    peak_u[8:540] = 0.20
    u = uniqueness_db("user", 0.08, 5.40, peak_u, peak_a, hop)
    a = uniqueness_db("assistant", 0.08, 5.40, peak_u, peak_a, hop)
    assert u > 20
    assert a < -20


def test_long_floor_kept_when_other_talks_inside():
    hop = 0.010
    n = 1200
    peak_u = np.zeros(n, dtype=np.float32)
    peak_a = np.zeros(n, dtype=np.float32)
    peak_u[100:1000] = 0.20
    peak_a[300:800] = 0.22
    clips = [
        ("user", 1.00, 10.00),
        ("assistant", 3.00, 8.00),
    ]
    kept, dropped = assign_script_turns(clips, peak_u, peak_a, hop)
    assert kept == [("user", 1.00, 10.00)]
    assert any(c[0] == "assistant" for c in dropped)


def test_padded_bleed_is_dropped_even_if_mean_is_close():
    """Quiet pickup plus pad used to average under the old -6 dB bar and survive."""
    hop = 0.010
    n = 800
    peak_u = np.zeros(n, dtype=np.float32)
    peak_a = np.zeros(n, dtype=np.float32)
    peak_u[100:500] = 0.20
    # Assistant clip is mostly pad; core is only ~4 dB down from the user.
    peak_a[80:250] = 0.002
    peak_a[100:180] = 0.13
    clips = [
        ("user", 1.00, 5.00),
        ("assistant", 0.80, 2.50),
    ]
    kept, dropped = assign_script_turns(clips, peak_u, peak_a, hop)
    assert kept == [("user", 1.00, 5.00)]
    assert any(c[0] == "assistant" for c in dropped)


def test_offset_bleed_after_other_stops_is_dropped():
    hop = 0.010
    n = 800
    peak_u = np.zeros(n, dtype=np.float32)
    peak_a = np.zeros(n, dtype=np.float32)
    peak_u[100:500] = 0.20
    # User already silent; leftover on assistant is much quieter than that turn.
    peak_a[505:580] = 0.04
    clips = [
        ("user", 1.00, 5.00),
        ("assistant", 5.05, 5.80),
    ]
    kept, dropped = assign_script_turns(clips, peak_u, peak_a, hop)
    assert kept == [("user", 1.00, 5.00)]
    assert any(c[0] == "assistant" for c in dropped)


def test_short_real_reply_at_similar_level_is_kept():
    hop = 0.010
    n = 900
    peak_u = np.zeros(n, dtype=np.float32)
    peak_a = np.zeros(n, dtype=np.float32)
    peak_u[100:500] = 0.20
    peak_a[525:620] = 0.18
    clips = [
        ("user", 1.00, 5.00),
        ("assistant", 5.25, 6.20),
    ]
    kept, dropped = assign_script_turns(clips, peak_u, peak_a, hop)
    assert any(c[0] == "assistant" and abs(c[1] - 5.25) < 1e-6 for c in kept)
    assert not any(c[0] == "assistant" for c in dropped)


def test_edge_overlap_bleed_dropped_even_when_median_looks_unique():
    """Bleed leftover: most frames are after the other mic stops, overlap is quieter."""
    hop = 0.010
    n = 900
    peak_u = np.zeros(n, dtype=np.float32)
    peak_a = np.zeros(n, dtype=np.float32)
    peak_u[100:500] = 0.20
    peak_a[470:560] = 0.05
    clips = [
        ("user", 1.00, 5.00),
        ("assistant", 4.70, 5.60),
    ]
    kept, dropped = assign_script_turns(clips, peak_u, peak_a, hop)
    assert kept == [("user", 1.00, 5.00)]
    assert any(c[0] == "assistant" for c in dropped)


def test_leading_quiet_assistant_before_user_is_dropped():
    hop = 0.010
    n = 2500
    peak_u = np.zeros(n, dtype=np.float32)
    peak_a = np.zeros(n, dtype=np.float32)
    peak_a[50:150] = 0.08
    peak_u[800:1500] = 0.20
    peak_a[1600:2400] = 0.20
    clips = [
        ("assistant", 0.50, 1.50),
        ("user", 8.00, 15.00),
        ("assistant", 16.00, 24.00),
    ]
    kept, dropped = assign_script_turns(clips, peak_u, peak_a, hop)
    assert not any(c[0] == "assistant" and c[2] < 8.0 for c in kept)
    assert any(c[0] == "assistant" and c[1] >= 16.0 for c in kept)
    assert any(c[0] == "assistant" and c[2] < 8.0 for c in dropped)


def test_leading_full_level_assistant_before_user_is_kept():
    hop = 0.010
    n = 2500
    peak_u = np.zeros(n, dtype=np.float32)
    peak_a = np.zeros(n, dtype=np.float32)
    peak_a[50:150] = 0.20
    peak_u[800:1500] = 0.20
    peak_a[1600:2400] = 0.20
    clips = [
        ("assistant", 0.50, 1.50),
        ("user", 8.00, 15.00),
        ("assistant", 16.00, 24.00),
    ]
    kept, dropped = assign_script_turns(clips, peak_u, peak_a, hop)
    assert any(c[0] == "assistant" and abs(c[1] - 0.50) < 1e-6 for c in kept)
