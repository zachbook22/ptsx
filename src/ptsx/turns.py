"""Exclusive-floor turn assignment with backchannel suppression."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ptsx.vad import mask_to_segments


@dataclass
class Cut:
    start_s: float
    end_s: float
    speaker: str
    reason: str

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_s - self.start_s)


@dataclass
class TurnParams:
    max_backchannel_s: float = 0.55
    pre_floor_s: float = 0.25
    post_hold_s: float = 0.40
    min_gap_s: float = 0.300
    max_gap_s: float = 0.800
    target_gap_s: float = 0.500
    min_core_s: float = 0.080
    hang_s: float = 0.35
    smooth_s: float = 0.16
    steal_ratio: float = 1.5
    min_long_turn_s: float = 0.70
    max_one_word_s: float = 0.60
    tail_hang_s: float = 0.080


@dataclass
class GateResult:
    keep_user: np.ndarray
    keep_asst: np.ndarray
    cuts: list[Cut] = field(default_factory=list)
    hop_s: float = 0.032


def assign_turns(
    user_speech: np.ndarray,
    asst_speech: np.ndarray,
    hop_s: float,
    params: TurnParams | None = None,
    user_rms: np.ndarray | None = None,
    asst_rms: np.ndarray | None = None,
    user_prob: np.ndarray | None = None,
    asst_prob: np.ndarray | None = None,
    user_f0: np.ndarray | None = None,
    asst_f0: np.ndarray | None = None,
    user_period: np.ndarray | None = None,
    asst_period: np.ndarray | None = None,
) -> GateResult:
    params = params or TurnParams()
    u_orig = np.asarray(user_speech, dtype=bool)
    a_orig = np.asarray(asst_speech, dtype=bool)
    n = min(len(u_orig), len(a_orig))
    u_orig, a_orig = u_orig[:n], a_orig[:n]
    ru = None if user_rms is None else np.asarray(user_rms, dtype=np.float32)[:n]
    ra = None if asst_rms is None else np.asarray(asst_rms, dtype=np.float32)[:n]
    up = None if user_prob is None else np.asarray(user_prob, dtype=np.float32)[:n]
    ap = None if asst_prob is None else np.asarray(asst_prob, dtype=np.float32)[:n]
    uf0 = None if user_f0 is None else np.asarray(user_f0, dtype=np.float32)[:n]
    af0 = None if asst_f0 is None else np.asarray(asst_f0, dtype=np.float32)[:n]
    us = None if user_period is None else np.asarray(user_period, dtype=np.float32)[:n]
    as_ = None if asst_period is None else np.asarray(asst_period, dtype=np.float32)[:n]

    u_uniq, a_uniq = _unique_speech(u_orig, a_orig, hop_s, params, ru, ra, up, ap, uf0, af0, us, as_)
    keep_u, keep_a, cuts = _drop_backchannels(u_uniq, a_uniq, hop_s, params)
    keep_u, keep_a, overlap_cuts = _exclusive_floor(
        keep_u, keep_a, hop_s, params, ru, ra
    )
    cuts.extend(overlap_cuts)
    keep_u, keep_a = _fill_exclusive_holes(keep_u, keep_a, hop_s, 0.36)
    keep_u, keep_a, interstitial = _drop_interstitial_one_words(
        keep_u, keep_a, hop_s, params
    )
    cuts.extend(interstitial)
    keep_u, keep_a = _extend_tails(keep_u, keep_a, hop_s, params.tail_hang_s)
    keep_u, keep_a, gap_cuts = enforce_gaps(keep_u, keep_a, hop_s, params)
    cuts.extend(gap_cuts)

    if np.any(keep_u & keep_a):
        keep_a = keep_a & ~keep_u

    return GateResult(
        keep_user=keep_u, keep_asst=keep_a, cuts=merge_cuts(cuts), hop_s=hop_s
    )


def _frames(seconds: float, hop_s: float) -> int:
    return max(1, int(round(seconds / hop_s)))


def _octave_related(f1: float, f2: float) -> bool:
    if f1 < 70.0 or f2 < 70.0:
        return False
    ratio = max(f1, f2) / min(f1, f2)
    return min(abs(ratio - 1.0), abs(ratio - 1.5), abs(ratio - 2.0), abs(ratio - 3.0)) < 0.15


def _unique_speech(
    u_act: np.ndarray,
    a_act: np.ndarray,
    hop_s: float,
    params: TurnParams,
    ru: np.ndarray | None,
    ra: np.ndarray | None,
    up: np.ndarray | None,
    ap: np.ndarray | None,
    uf0: np.ndarray | None,
    af0: np.ndarray | None,
    us: np.ndarray | None,
    as_: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """True speech on each track, with bleed (same voice on both mics) removed."""
    n = len(u_act)
    u_uniq = u_act.copy()
    a_uniq = a_act.copy()
    if ru is None or ra is None:
        return u_uniq, a_uniq

    steal = params.steal_ratio
    both = u_uniq & a_uniq
    u_uniq[both & (ru < ra * steal) & (ra > ru)] = False
    a_uniq[both & (ra < ru * steal) & (ru >= ra)] = False

    if us is None or as_ is None or uf0 is None or af0 is None:
        return u_uniq, a_uniq

    u_voiced = (us >= 0.50) & (ru >= 0.005)
    a_voiced = (as_ >= 0.48) & (ra >= 0.0035)
    if ap is not None:
        a_voiced |= (ap >= 0.65) & (ra >= 0.0025) & (us < 0.45)

    u_uniq = u_voiced | (u_act & (ru >= ra) & (ru >= 0.008))
    a_uniq = a_voiced | (a_act & (ra > ru) & (ra >= 0.006) & (us < 0.50))

    for i in range(n):
        if not (u_voiced[i] and a_voiced[i]):
            continue
        if not _octave_related(float(uf0[i]), float(af0[i])):
            continue
        # Same talker on both mics (bleed). Never give this frame to the
        # assistant: the user is the majority speaker and far-end playback
        # is usually louder on their track. Assistant-only speech is the
        # frames where the user mic is not voiced.
        a_uniq[i] = False
        u_uniq[i] = True

    # A loud, periodic user mic is not an assistant turn.
    a_uniq &= ~((us >= 0.50) & (ru >= 0.008))
    return u_uniq, a_uniq


def _drop_backchannels(
    u_orig: np.ndarray,
    a_orig: np.ndarray,
    hop_s: float,
    params: TurnParams,
) -> tuple[np.ndarray, np.ndarray, list[Cut]]:
    keep_u = u_orig.copy()
    keep_a = a_orig.copy()
    cuts: list[Cut] = []
    pre = _frames(params.pre_floor_s, hop_s)
    post = _frames(params.post_hold_s, hop_s)

    for speaker, mine, orig_mine, orig_other in (
        ("user", keep_u, u_orig, a_orig),
        ("assistant", keep_a, a_orig, u_orig),
    ):
        for start, end in mask_to_segments(orig_mine):
            dur = (end - start) * hop_s
            if dur > params.max_backchannel_s:
                continue
            before = orig_other[max(0, start - pre) : start]
            during = orig_other[start:end]
            after = orig_other[end : min(len(orig_other), end + post)]
            had_floor = before.size > 0 and float(before.mean()) > 0.45
            talking_through = during.size > 0 and float(during.mean()) > 0.45
            continues = after.size > 0 and float(after.mean()) > 0.25
            unanswered = had_floor and talking_through and continues
            if not unanswered:
                # When unsure, keep. Only cut clear listener cues.
                continue
            mine[start:end] = False
            reason = "backchannel" if dur <= 0.45 else "unanswered_interruption"
            cuts.append(Cut(start * hop_s, end * hop_s, speaker, reason))
    return keep_u, keep_a, cuts


def _exclusive_floor(
    keep_u: np.ndarray,
    keep_a: np.ndarray,
    hop_s: float,
    params: TurnParams,
    user_rms: np.ndarray | None,
    asst_rms: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, list[Cut]]:
    """Hold the floor until the other person has a real turn. Mute yeahs/bleed."""
    n = len(keep_u)
    ru = np.asarray(user_rms, dtype=np.float32) if user_rms is not None else keep_u.astype(
        np.float32
    )
    ra = np.asarray(asst_rms, dtype=np.float32) if asst_rms is not None else keep_a.astype(
        np.float32
    )
    hang = _frames(params.hang_s, hop_s)
    look = _frames(0.40, hop_s)
    min_other = _frames(max(params.max_backchannel_s, 0.55), hop_s)
    release_u = _frames(0.50, hop_s)
    release_a = _frames(0.42, hop_s)
    out_u = np.zeros(n, dtype=bool)
    out_a = np.zeros(n, dtype=bool)
    floor = 0
    silent_n = 0
    cuts: list[Cut] = []
    muted_from: int | None = None
    muted_who: str | None = None

    def flush_mute(i: int) -> None:
        nonlocal muted_from, muted_who
        if muted_from is not None and muted_who is not None and i > muted_from:
            cuts.append(
                Cut(muted_from * hop_s, i * hop_s, muted_who, "backchannel")
            )
        muted_from, muted_who = None, None

    def future(mask: np.ndarray, i: int, width: int | None = None) -> int:
        j = min(n, i + (width or look))
        return int(np.sum(mask[i:j]))

    def pick_start(i: int) -> int:
        fu, fa = future(keep_u, i), future(keep_a, i)
        if fu == 0 and fa == 0:
            return 0
        if fu > 0 and fu >= fa * 0.5:
            return 1
        if fa >= min_other or (fa > fu and fa >= hang):
            return 2
        if keep_u[i] and not keep_a[i]:
            return 1
        if keep_a[i] and not keep_u[i]:
            return 2
        if keep_u[i] and keep_a[i]:
            return 1 if float(ru[i]) >= float(ra[i]) else 2
        return 0

    for i in range(n):
        u, a = bool(keep_u[i]), bool(keep_a[i])
        if floor == 0:
            floor = pick_start(i)
            silent_n = 0
        elif floor == 1:
            if u:
                silent_n = 0
            else:
                silent_n += 1
                other_real = a and silent_n >= hang and future(keep_a, i, min_other) >= min_other
                if other_real and future(keep_u, i) < hang:
                    floor = 2
                    silent_n = 0
                elif silent_n >= release_u:
                    floor = pick_start(i)
                    silent_n = 0
        elif floor == 2:
            if a:
                silent_n = 0
            else:
                silent_n += 1
                other_real = u and silent_n >= hang and future(keep_u, i, min_other) >= hang
                if other_real:
                    floor = 1
                    silent_n = 0
                elif silent_n >= release_a:
                    floor = pick_start(i)
                    silent_n = 0

        if floor == 1:
            if u:
                out_u[i] = True
            if a:
                if muted_who != "assistant":
                    flush_mute(i)
                    muted_from, muted_who = i, "assistant"
            else:
                flush_mute(i)
        elif floor == 2:
            if a:
                out_a[i] = True
            if u:
                if muted_who != "user":
                    flush_mute(i)
                    muted_from, muted_who = i, "user"
            else:
                flush_mute(i)
        else:
            flush_mute(i)

    flush_mute(n)
    return out_u, out_a, cuts


def _drop_interstitial_one_words(
    keep_u: np.ndarray,
    keep_a: np.ndarray,
    hop_s: float,
    params: TurnParams,
) -> tuple[np.ndarray, np.ndarray, list[Cut]]:
    """Cut only short one-word answers / noises that sit between longer turns."""
    segs: list[tuple[str, int, int]] = []
    for sp, mask in (("user", keep_u), ("assistant", keep_a)):
        for s, e in mask_to_segments(mask):
            segs.append((sp, s, e))
    segs.sort(key=lambda x: x[1])
    longs = [
        (sp, s, e)
        for sp, s, e in segs
        if (e - s) * hop_s >= params.min_long_turn_s
    ]
    cuts: list[Cut] = []
    out_u, out_a = keep_u.copy(), keep_a.copy()
    for sp, s, e in segs:
        dur = (e - s) * hop_s
        if dur > params.max_one_word_s:
            continue
        prev_long = any(end <= s for _, _start, end in longs)
        next_long = any(start >= e for _, start, _end in longs)
        if not (prev_long and next_long):
            continue
        if sp == "user":
            out_u[s:e] = False
        else:
            out_a[s:e] = False
        cuts.append(Cut(s * hop_s, e * hop_s, sp, "one_word"))
    return out_u, out_a, cuts


def _extend_tails(
    keep_u: np.ndarray, keep_a: np.ndarray, hop_s: float, hang_s: float
) -> tuple[np.ndarray, np.ndarray]:
    """Keep a little room after the last word so the fade is never on dialogue."""
    hang = _frames(hang_s, hop_s)
    out_u, out_a = keep_u.copy(), keep_a.copy()
    n = len(out_u)
    for mask, other in ((out_u, out_a), (out_a, out_u)):
        for s, e in mask_to_segments(mask.copy()):
            extra = 0
            while extra < hang and e + extra < n and not other[e + extra] and not mask[e + extra]:
                extra += 1
            if extra:
                mask[e : e + extra] = True
    return out_u, out_a


def _fill_exclusive_holes(
    keep_u: np.ndarray, keep_a: np.ndarray, hop_s: float, max_hole_s: float
) -> tuple[np.ndarray, np.ndarray]:
    max_f = _frames(max_hole_s, hop_s)
    keep_u = _fill_holes(keep_u, keep_a, max_f)
    keep_a = _fill_holes(keep_a, keep_u, max_f)
    return keep_u, keep_a


def _fill_holes(mine: np.ndarray, other: np.ndarray, max_f: int) -> np.ndarray:
    out = mine.copy()
    n = len(out)
    i = 0
    while i < n:
        if not out[i]:
            j = i
            while j < n and not out[j]:
                j += 1
            if i > 0 and j < n and (j - i) <= max_f and not np.any(other[i:j]):
                out[i:j] = True
            i = j
        else:
            i += 1
    return out


def merge_cuts(cuts: list[Cut], join_s: float = 0.08) -> list[Cut]:
    if not cuts:
        return []
    ordered = sorted(cuts, key=lambda c: (c.speaker, c.reason, c.start_s))
    merged: list[Cut] = []
    for cut in ordered:
        join = 0.0 if cut.reason == "turn_gap" else join_s
        if (
            merged
            and merged[-1].speaker == cut.speaker
            and merged[-1].reason == cut.reason
            and cut.start_s <= merged[-1].end_s + join
        ):
            merged[-1] = Cut(
                merged[-1].start_s,
                max(merged[-1].end_s, cut.end_s),
                cut.speaker,
                cut.reason,
            )
        else:
            merged.append(cut)
    merged.sort(key=lambda c: (c.start_s, c.end_s, c.speaker))
    return merged


def enforce_gaps(
    keep_u: np.ndarray,
    keep_a: np.ndarray,
    hop_s: float,
    params: TurnParams,
) -> tuple[np.ndarray, np.ndarray, list[Cut]]:
    segs: list[list] = []
    for sp, mask in (("user", keep_u), ("assistant", keep_a)):
        for s, e in mask_to_segments(mask):
            segs.append([s, e, sp])
    segs.sort(key=lambda x: x[0])

    min_f = _frames(params.min_gap_s, hop_s)
    min_core = _frames(params.min_core_s, hop_s)
    cuts: list[Cut] = []

    for i in range(len(segs) - 1):
        cur, nxt = segs[i], segs[i + 1]
        if cur[2] == nxt[2]:
            continue
        gap = nxt[0] - cur[1]
        if gap >= min_f:
            continue
        need = max(0, min_f - gap)
        if need <= 0:
            continue
        # Never trim the outgoing tail (last words). Mute incoming pad only.
        right_avail = max(0, (nxt[1] - nxt[0]) - min_core)
        trim_r = min(need, right_avail)
        nxt[0] += trim_r
        if nxt[0] < cur[1] + min_f:
            nxt[0] = min(nxt[1], cur[1] + min_f)
            if nxt[1] - nxt[0] < min_core:
                nxt[0] = nxt[1]
        gap_s = (nxt[0] - cur[1]) * hop_s
        if cur[1] < nxt[0] and params.min_gap_s - 0.05 <= gap_s <= params.max_gap_s + 0.15:
            cuts.append(Cut(cur[1] * hop_s, nxt[0] * hop_s, "both", "turn_gap"))

    n = len(keep_u)
    out_u = np.zeros(n, dtype=bool)
    out_a = np.zeros(n, dtype=bool)
    for s, e, sp in segs:
        if e <= s:
            continue
        s = max(0, min(n, s))
        e = max(0, min(n, e))
        if sp == "user":
            out_u[s:e] = True
        else:
            out_a[s:e] = True
    return out_u, out_a, cuts


def chunk_sample_bounds(
    keep_u: np.ndarray,
    keep_a: np.ndarray,
    hop_s: float,
    sr: int,
    n_samples: int,
    chunk_minutes: float | None,
) -> list[tuple[int, int]]:
    """Split at speaker-change gaps near each chunk_minutes mark. Full file if None."""
    if not chunk_minutes or chunk_minutes <= 0:
        return [(0, n_samples)]

    target = int(chunk_minutes * 60 * sr)
    if target >= n_samples:
        return [(0, n_samples)]

    segs: list[tuple[int, int, str]] = []
    for sp, mask in (("user", keep_u), ("assistant", keep_a)):
        for s, e in mask_to_segments(mask):
            segs.append((s, e, sp))
    segs.sort()

    gap_mids: list[int] = []
    for i in range(len(segs) - 1):
        if segs[i][2] == segs[i + 1][2]:
            continue
        gap_start = int(segs[i][1] * hop_s * sr)
        gap_end = int(segs[i + 1][0] * hop_s * sr)
        if gap_end > gap_start:
            gap_mids.append((gap_start + gap_end) // 2)

    bounds = [0]
    cursor = target
    while cursor < n_samples:
        later = [g for g in gap_mids if g >= cursor and g > bounds[-1]]
        if later:
            cut = later[0]
            if cut <= bounds[-1]:
                break
            bounds.append(cut)
            cursor = cut + target
        else:
            break
    if bounds[-1] != n_samples:
        bounds.append(n_samples)
    return list(zip(bounds[:-1], bounds[1:]))
