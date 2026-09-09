"""Linear conversation script: unique-speaker turns, never trimmed at either end."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ptsx.strip import _overlap, merge_near_clips


def mean_peak(
    peak: np.ndarray, start_s: float, end_s: float, hop_s: float
) -> float:
    i0 = max(0, int(start_s / hop_s))
    i1 = min(len(peak), int(np.ceil(end_s / hop_s)))
    if i1 <= i0:
        return 0.0
    return float(np.mean(peak[i0:i1]))


def uniqueness_db(
    speaker: str,
    start_s: float,
    end_s: float,
    peak_u: np.ndarray,
    peak_a: np.ndarray,
    hop_s: float,
) -> float:
    mine = mean_peak(peak_u if speaker == "user" else peak_a, start_s, end_s, hop_s)
    other = mean_peak(peak_a if speaker == "user" else peak_u, start_s, end_s, hop_s)
    return float(20.0 * np.log10((mine + 1e-12) / (other + 1e-12)))


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
    for sp, start, end in clips:
        uniq = uniqueness_db(sp, start, end, peak_u, peak_a, hop_s)
        if uniq < -unique_db:
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
        # Pad/hysteresis glue is not a nested turn. Keep both clips whole.
        if ov is None or ov_len <= 0.30:
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
    return turns, dropped


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
