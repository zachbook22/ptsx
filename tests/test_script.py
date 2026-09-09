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
