"""Place kept turns on a new timeline: never shorten a tail, nudge the next clip."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ptsx.turns import TurnParams
from ptsx.vad import mask_to_segments


@dataclass
class PlacedClip:
    speaker: str
    src_start_s: float
    src_end_s: float
    dst_start_s: float
    dst_end_s: float


def segments_from_masks(
    keep_u: np.ndarray, keep_a: np.ndarray, hop_s: float
) -> list[tuple[str, int, int]]:
    segs: list[tuple[str, int, int]] = []
    for sp, mask in (("user", keep_u), ("assistant", keep_a)):
        for s, e in mask_to_segments(mask):
            segs.append((sp, s, e))
    segs.sort(key=lambda x: x[1])
    return segs


def place_time_clips(
    clips: list[tuple[str, float, float]],
    params: TurnParams | None = None,
    dropped_s: list[tuple[float, float]] | None = None,
) -> list[PlacedClip]:
    """Lay kept source clips on a new timeline. Source in/out are never shortened.

    After strip silence, dead air is gone. Speaker-change gaps outside
    0.300–0.800 s become the target (default 0.500 s). Same-speaker holes
    bigger than max_gap are packed the same way.
    """
    params = params or TurnParams()
    ordered = sorted(clips, key=lambda c: (c[1], c[2], c[0]))
    placed: list[PlacedClip] = []
    dst = 0.0
    prev_sp: str | None = None
    prev_src_end = 0.0

    for sp, src0, src1 in ordered:
        if src1 <= src0:
            continue
        if prev_sp is None:
            gap = 0.0
        elif prev_sp == sp:
            natural = max(0.0, src0 - prev_src_end)
            if natural > params.max_gap_s:
                gap = params.target_gap_s
            else:
                gap = natural
        else:
            natural = src0 - prev_src_end
            if natural < params.min_gap_s or natural > params.max_gap_s:
                gap = params.target_gap_s
            else:
                gap = float(max(natural, 0.0))
        dst += gap
        length = src1 - src0
        placed.append(
            PlacedClip(
                speaker=sp,
                src_start_s=src0,
                src_end_s=src1,
                dst_start_s=dst,
                dst_end_s=dst + length,
            )
        )
        dst += length
        prev_sp = sp
        prev_src_end = src1
    return placed


def place_segments(
    segs: list[tuple[str, int, int]],
    hop_s: float,
    params: TurnParams | None = None,
    dropped_s: list[tuple[float, float]] | None = None,
) -> list[PlacedClip]:
    clips = [(sp, s * hop_s, e * hop_s) for sp, s, e in segs]
    return place_time_clips(clips, params, dropped_s)


def render_placed(
    user: np.ndarray,
    asst: np.ndarray,
    clips: list[PlacedClip],
    sr: int,
    fade_ms: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Copy each source clip in full. Fades live on quiet pad, never on words."""
    fade_n = max(1, int(round(sr * fade_ms / 1000.0)))
    if not clips:
        empty = np.zeros(0, dtype=np.float32)
        return empty, empty, empty
    n_out = int(round(clips[-1].dst_end_s * sr)) + fade_n + 1
    n_in = min(len(user), len(asst))
    user_out = np.zeros(n_out, dtype=np.float32)
    asst_out = np.zeros(n_out, dtype=np.float32)
    speech_thresh = 0.02

    for clip in clips:
        src0 = int(round(clip.src_start_s * sr))
        src1 = int(round(clip.src_end_s * sr))
        dst0 = int(round(clip.dst_start_s * sr))
        src0 = max(0, min(n_in, src0))
        src1 = max(src0, min(n_in, src1))
        if src1 <= src0:
            continue
        raw = user if clip.speaker == "user" else asst
        dest = user_out if clip.speaker == "user" else asst_out

        pre0 = max(0, src0 - fade_n)
        pre = raw[pre0:src0].copy()
        if pre.size < fade_n:
            pre = np.pad(pre, (fade_n - pre.size, 0))
        pre = pre[-fade_n:].astype(np.float32, copy=False)
        if float(np.max(np.abs(pre))) > speech_thresh:
            pre = np.zeros(fade_n, dtype=np.float32)
        pre = pre * np.linspace(0.0, 1.0, fade_n, dtype=np.float32)

        post1 = min(n_in, src1 + fade_n)
        post = raw[src1:post1].copy()
        if post.size < fade_n:
            post = np.pad(post, (0, fade_n - post.size))
        post = post[:fade_n].astype(np.float32, copy=False)
        if float(np.max(np.abs(post))) > speech_thresh:
            post = np.zeros(fade_n, dtype=np.float32)
        post = post * np.linspace(1.0, 0.0, fade_n, dtype=np.float32)

        chunk = np.concatenate([pre, raw[src0:src1], post])
        write0 = max(0, dst0 - fade_n)
        write1 = min(n_out, write0 + len(chunk))
        dest[write0:write1] += chunk[: write1 - write0]

    both = (np.abs(user_out) > 0.0) & (np.abs(asst_out) > 0.0)
    if np.any(both):
        prefer_u = np.abs(user_out) >= np.abs(asst_out)
        asst_out[both & prefer_u] = 0.0
        user_out[both & ~prefer_u] = 0.0
    return user_out, asst_out, user_out + asst_out


def removed_stems(
    user: np.ndarray,
    asst: np.ndarray,
    clips: list[PlacedClip],
    sr: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Same length as the raws: kept source regions zeroed, dropped audio left."""
    n = min(len(user), len(asst))
    u_gain = np.ones(n, dtype=np.float32)
    a_gain = np.ones(n, dtype=np.float32)
    for clip in clips:
        i0 = max(0, min(n, int(round(clip.src_start_s * sr))))
        i1 = max(0, min(n, int(round(clip.src_end_s * sr))))
        if clip.speaker == "user":
            u_gain[i0:i1] = 0.0
        else:
            a_gain[i0:i1] = 0.0
    return user[:n] * u_gain, asst[:n] * a_gain
