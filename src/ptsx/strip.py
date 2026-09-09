"""Pro Tools-style strip silence, then exclusive clips for nudging."""

from __future__ import annotations

import numpy as np

from ptsx.vad import mask_to_segments


def db_to_amp(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def frame_peak(audio: np.ndarray, sr: int, hop_s: float) -> np.ndarray:
    hop = max(1, int(round(sr * hop_s)))
    n = len(audio) // hop
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    frames = np.abs(audio[: n * hop]).reshape(n, hop)
    return frames.max(axis=1).astype(np.float32)


def auto_hold_db(
    audio: np.ndarray,
    sr: int,
    hop_s: float = 0.010,
    thresh_db: float = -28.0,
) -> float:
    """Hold above this track's room tone so a noisy mic is not one endless clip."""
    peak = frame_peak(audio, sr, hop_s)
    if peak.size == 0:
        return thresh_db - 8.0
    noise_db = float(20.0 * np.log10(float(np.percentile(peak, 40)) + 1e-12))
    return float(max(thresh_db - 8.0, noise_db + 3.0))


def strip_silence(
    audio: np.ndarray,
    sr: int,
    thresh_db: float = -28.0,
    pad_s: float = 0.250,
    hop_s: float = 0.010,
    min_keep_s: float = 0.040,
    hold_db: float | None = None,
    min_silence_s: float = 0.200,
) -> np.ndarray:
    """Keep mask at `hop_s`: open at threshold, hold through quiet word tails, then pad."""
    peak = frame_peak(audio, sr, hop_s)
    if peak.size == 0:
        return np.zeros(0, dtype=bool)
    open_amp = db_to_amp(thresh_db)
    close_amp = db_to_amp(thresh_db - 8.0 if hold_db is None else hold_db)
    min_sil = max(1, int(round(min_silence_s / hop_s)))
    keep = _hysteresis_keep(peak, open_amp, close_amp, min_sil)
    min_keep = max(1, int(round(min_keep_s / hop_s)))
    keep = _drop_short(keep, min_keep)
    pad = max(0, int(round(pad_s / hop_s)))
    if pad:
        keep = _pad_segments(keep, pad)
    return keep


def _hysteresis_keep(
    peak: np.ndarray, open_amp: float, close_amp: float, min_sil: int
) -> np.ndarray:
    """Stay on through tails below the open threshold until real silence."""
    n = len(peak)
    keep = np.zeros(n, dtype=bool)
    on = False
    below = 0
    for i, p in enumerate(peak):
        if not on:
            if p >= open_amp:
                on = True
                below = 0
                keep[i] = True
            continue
        if p >= close_amp:
            keep[i] = True
            below = 0
            continue
        below += 1
        keep[i] = True
        if below >= min_sil:
            keep[i - min_sil + 1 : i + 1] = False
            on = False
            below = 0
    return keep


def _drop_short(mask: np.ndarray, min_len: int) -> np.ndarray:
    out = mask.copy()
    n = len(out)
    i = 0
    while i < n:
        if out[i]:
            j = i
            while j < n and out[j]:
                j += 1
            if j - i < min_len:
                out[i:j] = False
            i = j
        else:
            i += 1
    return out


def _pad_segments(mask: np.ndarray, pad: int) -> np.ndarray:
    n = len(mask)
    out = mask.copy()
    for start, end in mask_to_segments(mask):
        out[max(0, start - pad) : min(n, end + pad)] = True
    return out


def exclusive_stripped(
    keep_u: np.ndarray,
    keep_a: np.ndarray,
    peak_u: np.ndarray,
    peak_a: np.ndarray,
    hop_s: float,
    hang_s: float = 0.200,
) -> tuple[np.ndarray, np.ndarray]:
    """Inside padded strip regions, only one speaker. Hang avoids chatter."""
    n = min(len(keep_u), len(keep_a), len(peak_u), len(peak_a))
    keep_u = np.asarray(keep_u[:n], dtype=bool)
    keep_a = np.asarray(keep_a[:n], dtype=bool)
    peak_u = np.asarray(peak_u[:n], dtype=np.float32)
    peak_a = np.asarray(peak_a[:n], dtype=np.float32)
    hang = max(1, int(round(hang_s / hop_s)))
    out_u = np.zeros(n, dtype=bool)
    out_a = np.zeros(n, dtype=bool)
    floor = 0
    pend = 0
    pend_who = 0

    def want_at(i: int) -> int:
        u, a = bool(keep_u[i]), bool(keep_a[i])
        if u and not a:
            return 1
        if a and not u:
            return 2
        if u and a:
            return 1 if peak_u[i] >= peak_a[i] else 2
        return 0

    for i in range(n):
        want = want_at(i)
        if floor == 0:
            floor = want
            pend = 0
        elif want == floor:
            pend = 0
        elif want == 0:
            # Strip ended for this speaker; do not hold through silence.
            floor = 0
            pend = 0
        else:
            if pend_who != want:
                pend_who = want
                pend = 1
            else:
                pend += 1
            if pend >= hang:
                floor = want
                pend = 0

        if floor == 1:
            out_u[i] = True
        elif floor == 2:
            out_a[i] = True

    return out_u, out_a


def clips_from_mask(mask: np.ndarray, hop_s: float, speaker: str) -> list[tuple[str, float, float]]:
    return [(speaker, s * hop_s, e * hop_s) for s, e in mask_to_segments(mask)]


def merge_near_clips(
    clips: list[tuple[str, float, float]], max_gap_s: float = 0.30
) -> list[tuple[str, float, float]]:
    """Join same-speaker clips whose pads almost touch so punch doesn't leave a sliver."""
    by_sp: dict[str, list[list[float]]] = {}
    for sp, start, end in clips:
        by_sp.setdefault(sp, []).append([start, end])
    out: list[tuple[str, float, float]] = []
    for sp, items in by_sp.items():
        items.sort()
        merged: list[list[float]] = []
        for start, end in items:
            if merged and start - merged[-1][1] <= max_gap_s:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        out.extend((sp, a, b) for a, b in merged)
    out.sort(key=lambda c: (c[1], c[2], c[0]))
    return out


def clips_to_mask(
    clips: list[tuple[str, float, float]],
    hop_s: float,
    n_frames: int,
    speaker: str,
) -> np.ndarray:
    mask = np.zeros(n_frames, dtype=bool)
    for sp, start, end in clips:
        if sp != speaker:
            continue
        i0 = max(0, int(round(start / hop_s)))
        i1 = min(n_frames, int(round(end / hop_s)))
        if i1 > i0:
            mask[i0:i1] = True
    return mask


def _overlap(a0: float, a1: float, b0: float, b1: float) -> tuple[float, float] | None:
    o0, o1 = max(a0, b0), min(a1, b1)
    if o1 - o0 <= 1e-6:
        return None
    return o0, o1


def _subtract(start: float, end: float, o0: float, o1: float) -> list[tuple[float, float]]:
    """Remove [o0, o1] from [start, end]; return remaining pieces."""
    pieces: list[tuple[float, float]] = []
    if o0 > start + 1e-6:
        pieces.append((start, min(end, o0)))
    if o1 < end - 1e-6:
        pieces.append((max(start, o1), end))
    return [(a, b) for a, b in pieces if b - a > 1e-6]


def resolve_nested_clips(
    clips: list[tuple[str, float, float]],
    peak_u: np.ndarray,
    peak_a: np.ndarray,
    hop_s: float,
    max_backchannel_s: float = 0.70,
) -> tuple[list[tuple[str, float, float]], list[tuple[str, float, float]]]:
    """When a long clip covers a shorter other-speaker clip, drop bleed/yeahs
    or punch a hole in the sprawling (bleed) clip so the tighter turn remains.
    """
    items: list[list] = [[sp, s, e] for sp, s, e in clips]
    dropped: list[tuple[str, float, float]] = []

    def mean_peak(sp: str, t0: float, t1: float) -> float:
        peak = peak_u if sp == "user" else peak_a
        i0 = max(0, int(t0 / hop_s))
        i1 = min(len(peak), int(np.ceil(t1 / hop_s)))
        if i1 <= i0:
            return 0.0
        return float(np.mean(peak[i0:i1]))

    changed = True
    while changed:
        changed = False
        items = [c for c in items if c[2] - c[1] > 1e-4]
        items.sort(key=lambda c: (c[1], c[2], c[0]))
        pair = None
        for i, a in enumerate(items):
            for j, b in enumerate(items):
                if j <= i or a[0] == b[0]:
                    continue
                ov = _overlap(a[1], a[2], b[1], b[2])
                if ov is None:
                    continue
                pair = (i, j, a, b, ov)
                break
            if pair is not None:
                break
        if pair is None:
            break
        i, j, a, b, (o0, o1) = pair
        ov_len = o1 - o0
        len_a, len_b = a[2] - a[1], b[2] - b[1]
        frac_a = ov_len / max(len_a, 1e-6)
        frac_b = ov_len / max(len_b, 1e-6)

        def drop(idx: int) -> None:
            sp, s, e = items[idx]
            dropped.append((sp, s, e))
            del items[idx]

        def punch(idx: int, x0: float, x1: float) -> None:
            sp, s, e = items[idx]
            pieces = _subtract(s, e, x0, x1)
            del items[idx]
            for p0, p1 in reversed(pieces):
                items.insert(idx, [sp, p0, p1])
            dropped.append((sp, x0, x1))

        # B is nested in A (or vice versa): sprawling clip is usually bleed.
        nested_b = frac_b >= 0.65 and len_a >= len_b + 0.35
        nested_a = frac_a >= 0.65 and len_b >= len_a + 0.35
        if nested_b:
            if len_b <= max_backchannel_s:
                drop(j)
            else:
                punch(i, o0, o1)
            changed = True
            continue
        if nested_a:
            if len_a <= max_backchannel_s:
                drop(i)
            else:
                punch(j, o0, o1)
            changed = True
            continue

        # Partial overlap: louder side keeps the overlap.
        if mean_peak(a[0], o0, o1) >= mean_peak(b[0], o0, o1):
            punch(j, o0, o1)
        else:
            punch(i, o0, o1)
        changed = True

    kept = [(sp, s, e) for sp, s, e in items if e - s >= 0.08]
    return merge_near_clips(kept, max_gap_s=0.30), dropped
