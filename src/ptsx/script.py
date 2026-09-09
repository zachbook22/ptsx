"""Linear conversation script: unique-speaker turns, never trimmed at either end."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ptsx.strip import _overlap, merge_near_clips

# Ignore digital hush so pad/silence frames do not pull uniqueness toward 0 dB.
_ACTIVE_AMP = float(10.0 ** (-50.0 / 20.0))
# Other mic is "in speech" for overlap-bleed (above typical strip hold).
_OTHER_SPEECH_AMP = float(10.0 ** (-40.0 / 20.0))
_MIN_OTHER_SPEECH_FRAC = 0.12
_SHORT_BLEED_S = 2.60


def _peak_slice(
    peak: np.ndarray, start_s: float, end_s: float, hop_s: float
) -> np.ndarray:
    i0 = max(0, int(start_s / hop_s))
    i1 = min(len(peak), int(np.ceil(end_s / hop_s)))
    if i1 <= i0:
        return np.zeros(0, dtype=np.float32)
    return np.asarray(peak[i0:i1], dtype=np.float32)


def mean_peak(
    peak: np.ndarray, start_s: float, end_s: float, hop_s: float
) -> float:
    frames = _peak_slice(peak, start_s, end_s, hop_s)
    if frames.size == 0:
        return 0.0
    return float(np.mean(frames))


def clip_level_db(
    peak: np.ndarray, start_s: float, end_s: float, hop_s: float
) -> float:
    """Typical speech level in a clip (median of frames above the hush floor)."""
    frames = _peak_slice(peak, start_s, end_s, hop_s)
    if frames.size == 0:
        return -120.0
    active = frames >= _ACTIVE_AMP
    amp = float(np.median(frames[active] if np.any(active) else frames))
    return float(20.0 * np.log10(amp + 1e-12))


def uniqueness_db(
    speaker: str,
    start_s: float,
    end_s: float,
    peak_u: np.ndarray,
    peak_a: np.ndarray,
    hop_s: float,
) -> float:
    """Median per-frame dB of this mic vs the other. Pad hush is ignored."""
    mine = _peak_slice(peak_u if speaker == "user" else peak_a, start_s, end_s, hop_s)
    other = _peak_slice(peak_a if speaker == "user" else peak_u, start_s, end_s, hop_s)
    n = min(len(mine), len(other))
    if n <= 0:
        return 0.0
    mine, other = mine[:n], other[:n]
    active = np.maximum(mine, other) >= _ACTIVE_AMP
    if not np.any(active):
        return float(20.0 * np.log10((float(np.mean(mine)) + 1e-12) / (float(np.mean(other)) + 1e-12)))
    return float(
        np.median(
            20.0 * np.log10((mine[active] + 1e-12) / (other[active] + 1e-12))
        )
    )


def other_speech_uniqueness(
    speaker: str,
    start_s: float,
    end_s: float,
    peak_u: np.ndarray,
    peak_a: np.ndarray,
    hop_s: float,
    min_frac: float = _MIN_OTHER_SPEECH_FRAC,
    pad_s: float = 0.15,
) -> float:
    """Median dB of this mic vs the other, only on frames where the other is in speech.

    Edge bleed looks unique if the leftover sits after the other person stops.
    Those clips still lose on the overlapping frames. A little pad catches
    bleed that lives on the clip boundary. Returns +99 if the other mic is
    not in speech for enough of the window.
    """
    mine = _peak_slice(
        peak_u if speaker == "user" else peak_a,
        start_s - pad_s,
        end_s + pad_s,
        hop_s,
    )
    other = _peak_slice(
        peak_a if speaker == "user" else peak_u,
        start_s - pad_s,
        end_s + pad_s,
        hop_s,
    )
    n = min(len(mine), len(other))
    if n <= 0:
        return 99.0
    mine, other = mine[:n], other[:n]
    mask = other >= _OTHER_SPEECH_AMP
    if float(np.mean(mask)) < min_frac:
        return 99.0
    return float(
        np.median(20.0 * np.log10((mine[mask] + 1e-12) / (other[mask] + 1e-12)))
    )


def _typical_clip_db(
    clips: list[tuple[str, float, float]],
    speaker: str,
    peak_u: np.ndarray,
    peak_a: np.ndarray,
    hop_s: float,
    min_s: float = 3.0,
    unique_db: float = 6.0,
) -> float | None:
    peak = peak_u if speaker == "user" else peak_a
    levels: list[float] = []
    for sp, start, end in clips:
        if sp != speaker or end - start < min_s:
            continue
        if uniqueness_db(sp, start, end, peak_u, peak_a, hop_s) < unique_db:
            continue
        levels.append(clip_level_db(peak, start, end, hop_s))
    if not levels:
        return None
    return float(np.median(np.asarray(levels, dtype=np.float64)))


def _first_start(clips: list[tuple[str, float, float]], speaker: str) -> float | None:
    times = [start for sp, start, _end in clips if sp == speaker]
    return min(times) if times else None


def assign_script_turns(
    clips: list[tuple[str, float, float]],
    peak_u: np.ndarray,
    peak_a: np.ndarray,
    hop_s: float,
    unique_db: float = 6.0,
    max_backchannel_s: float = 0.70,
    min_tail_s: float = 0.70,
    merge_gap_s: float = 0.50,
) -> tuple[list[tuple[str, float, float]], list[tuple[str, float, float]]]:
    """Keep full unique turns. Nested yeahs/bleed are dropped; floors are never punched."""
    clips = merge_near_clips(clips, max_gap_s=merge_gap_s)
    dropped: list[tuple[str, float, float]] = []
    cand: list[list] = []
    typical = {
        "user": _typical_clip_db(clips, "user", peak_u, peak_a, hop_s),
        "assistant": _typical_clip_db(clips, "assistant", peak_u, peak_a, hop_s),
    }
    first = {
        "user": _first_start(clips, "user"),
        "assistant": _first_start(clips, "assistant"),
    }
    for sp, start, end in clips:
        uniq = uniqueness_db(sp, start, end, peak_u, peak_a, hop_s)
        dur = end - start
        other = "assistant" if sp == "user" else "user"
        # Long floors can contain the other person; only dump a clip that is
        # clearly quieter overall.
        if uniq < -unique_db:
            dropped.append((sp, start, end))
            continue
        if dur <= _SHORT_BLEED_S:
            # Leftover pickup: unique after the other mic stops, quieter
            # while it is still in speech.
            if other_speech_uniqueness(sp, start, end, peak_u, peak_a, hop_s) < 0.0:
                dropped.append((sp, start, end))
                continue
            first_other = first[other]
            typical_db = typical[sp]
            peak = peak_u if sp == "user" else peak_a
            if (
                first_other is not None
                and end <= first_other
                and typical_db is not None
                and clip_level_db(peak, start, end, hop_s) < typical_db
            ):
                dropped.append((sp, start, end))
                continue
        cand.append([sp, start, end, uniq])
    cand.sort(key=lambda c: (c[1], c[2], c[0]))

    kept: list[list] = []
    for sp, start, end, uniq in cand:
        if not kept:
            kept.append([sp, start, end, uniq])
            continue
        floor = kept[-1]
        if sp == floor[0]:
            if start <= floor[2] + merge_gap_s:
                floor[2] = max(floor[2], end)
                floor[3] = max(floor[3], uniq)
            else:
                kept.append([sp, start, end, uniq])
            continue
        ov = _overlap(floor[1], floor[2], start, end)
        ov_len = 0.0 if ov is None else ov[1] - ov[0]
        # Pad/hysteresis glue is not a nested turn. Keep both clips whole
        # unless this one is a quiet leftover on the other mic.
        if ov is None or ov_len <= 0.30:
            if _edge_bleed_vs(
                sp,
                start,
                end,
                floor[0],
                floor[1],
                floor[2],
                peak_u,
                peak_a,
                hop_s,
                unique_db,
            ):
                dropped.append((sp, start, end))
                continue
            kept.append([sp, start, end, uniq])
            continue
        dur = end - start
        tail = end - floor[2]
        if dur <= max_backchannel_s:
            dropped.append((sp, start, end))
            continue
        if tail >= min_tail_s and uniq >= unique_db:
            # Incoming pad yields to the floor tail; the new turn still keeps its ending.
            new_start = max(start, floor[2])
            if end - new_start >= min_tail_s:
                kept.append([sp, new_start, end, uniq])
            else:
                dropped.append((sp, start, end))
        else:
            dropped.append((sp, start, end))

    turns = [(sp, s, e) for sp, s, e, _ in kept if e - s >= 0.08]
    turns, glue_dropped = drop_glue_backchannels(
        turns, max_s=max(max_backchannel_s, 0.85)
    )
    dropped.extend(glue_dropped)
    turns, edge_dropped = drop_edge_bleed(
        turns, peak_u, peak_a, hop_s, quieter_db=unique_db
    )
    dropped.extend(edge_dropped)
    return turns, dropped


def _interval_gap(a0: float, a1: float, b0: float, b1: float) -> float:
    if _overlap(a0, a1, b0, b1) is not None:
        return 0.0
    if b0 >= a1:
        return b0 - a1
    return a0 - b1


def _edge_bleed_vs(
    speaker: str,
    start_s: float,
    end_s: float,
    other_sp: str,
    other_start: float,
    other_end: float,
    peak_u: np.ndarray,
    peak_a: np.ndarray,
    hop_s: float,
    quieter_db: float,
    max_s: float = 2.60,
    abut_s: float = 0.40,
) -> bool:
    """True when a short clip is much quieter than the neighboring other-speaker turn."""
    if speaker == other_sp:
        return False
    if end_s - start_s > max_s:
        return False
    if _interval_gap(start_s, end_s, other_start, other_end) > abut_s:
        return False
    mine = clip_level_db(
        peak_u if speaker == "user" else peak_a, start_s, end_s, hop_s
    )
    neigh = clip_level_db(
        peak_u if other_sp == "user" else peak_a,
        other_start,
        other_end,
        hop_s,
    )
    return mine - neigh < -quieter_db


def drop_edge_bleed(
    turns: list[tuple[str, float, float]],
    peak_u: np.ndarray,
    peak_a: np.ndarray,
    hop_s: float,
    quieter_db: float = 6.0,
    max_s: float = 2.60,
    abut_s: float = 0.40,
) -> tuple[list[tuple[str, float, float]], list[tuple[str, float, float]]]:
    """Mute small other-mic leftovers that sit on a speaker change (onset/offset bleed)."""
    dropped: list[tuple[str, float, float]] = []
    flags = [True] * len(turns)
    for i, (sp, start, end) in enumerate(turns):
        if end - start > max_s:
            continue
        for j, (osp, os, oe) in enumerate(turns):
            if j == i:
                continue
            if _edge_bleed_vs(
                sp,
                start,
                end,
                osp,
                os,
                oe,
                peak_u,
                peak_a,
                hop_s,
                quieter_db,
                max_s=max_s,
                abut_s=abut_s,
            ):
                flags[i] = False
                dropped.append((sp, start, end))
                break
    kept = [t for t, keep in zip(turns, flags) if keep]
    return kept, dropped


def drop_glue_backchannels(
    turns: list[tuple[str, float, float]],
    max_s: float = 0.70,
    glue_s: float = 0.20,
) -> tuple[list[tuple[str, float, float]], list[tuple[str, float, float]]]:
    """Mute a short mm-hmm that starts on the previous speaker's tail. Keep answered yeahs."""
    kept: list[tuple[str, float, float]] = []
    dropped: list[tuple[str, float, float]] = []
    for sp, start, end in turns:
        dur = end - start
        if (
            kept
            and dur <= max_s
            and kept[-1][0] != sp
            and start - kept[-1][2] <= glue_s
        ):
            dropped.append((sp, start, end))
            continue
        kept.append((sp, start, end))
    return kept, dropped


def overlay_transcript(
    turns: list[tuple[str, float, float]],
    user_segs: list[dict],
    asst_segs: list[dict],
) -> list[tuple[str, float, float, str]]:
    """Attach same-speaker transcript text whose midpoint falls inside each turn."""
    by_sp = {"user": user_segs, "assistant": asst_segs}
    out: list[tuple[str, float, float, str]] = []
    for sp, start, end in turns:
        bits: list[str] = []
        for seg in by_sp.get(sp, []):
            s0 = float(seg["start"])
            s1 = float(seg["end"])
            text = str(seg.get("text", "")).strip()
            if not text:
                continue
            ov0, ov1 = max(start, s0), min(end, s1)
            if ov1 - ov0 <= 0.05:
                continue
            bits.append(text)
        out.append((sp, start, end, " ".join(bits)))
    return out


def write_script_txt(
    path: Path, rows: list[tuple[str, float, float, str]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Conversation script (unique turns, original timeline)", ""]
    for i, (sp, start, end, text) in enumerate(rows, 1):
        label = "User" if sp == "user" else "Assistant"
        body = text.strip() if text.strip() else "(no transcript)"
        lines.append(f"{i:02d}. [{start:7.2f}–{end:7.2f}] {label}: {body}")
    path.write_text("\n".join(lines) + "\n")


def write_script_csv(
    path: Path, rows: list[tuple[str, float, float, str]]
) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n", "speaker", "start_s", "end_s", "text"])
        for i, (sp, start, end, text) in enumerate(rows, 1):
            w.writerow([i, sp, f"{start:.3f}", f"{end:.3f}", text.strip()])


def turns_from_clips_csv(path: Path) -> list[tuple[str, float, float]]:
    import csv

    rows: list[tuple[str, float, float]] = []
    with Path(path).open(newline="") as f:
        for r in csv.DictReader(f):
            rows.append(
                (r["speaker"], float(r["orig_start_s"]), float(r["orig_end_s"]))
            )
    return rows
