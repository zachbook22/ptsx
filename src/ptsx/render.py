"""Apply fades, write gated / removed stems, and the cut log."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from ptsx.audio import write_wav_24
from ptsx.turns import Cut
from ptsx.vad import mask_to_segments


def timecode(seconds: float) -> str:
    ms = int(round(max(0.0, seconds) * 1000.0))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def keep_mask_to_gain(
    mask: np.ndarray,
    hop_s: float,
    sr: int,
    n_samples: int,
    fade_ms: float = 10.0,
) -> np.ndarray:
    gain = np.zeros(n_samples, dtype=np.float32)
    fade_n = max(1, int(round(sr * fade_ms / 1000.0)))
    for start, end in mask_to_segments(mask):
        i0 = int(round(start * hop_s * sr))
        i1 = int(round(end * hop_s * sr))
        i0 = max(0, min(n_samples, i0))
        i1 = max(0, min(n_samples, i1))
        if i1 <= i0:
            continue
        gain[i0:i1] = 1.0
        fade = min(fade_n, max(1, (i1 - i0) // 2))
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        gain[i0 : i0 + fade] *= ramp
        gain[i1 - fade : i1] *= ramp[::-1]
    return gain


def exclusive_gains(gain_u: np.ndarray, gain_a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    both = (gain_u > 0.0) & (gain_a > 0.0)
    if not np.any(both):
        return gain_u, gain_a
    prefer_u = gain_u >= gain_a
    gain_a = gain_a.copy()
    gain_u = gain_u.copy()
    gain_a[both & prefer_u] = 0.0
    gain_u[both & ~prefer_u] = 0.0
    return gain_u, gain_a


def render_stems(
    user: np.ndarray,
    asst: np.ndarray,
    gain_u: np.ndarray,
    gain_a: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    user_out = user * gain_u
    asst_out = asst * gain_a
    user_removed = user * (1.0 - gain_u)
    asst_removed = asst * (1.0 - gain_a)
    combined = user_out + asst_out
    return user_out, asst_out, combined, user_removed, asst_removed


def write_cuts_csv(path: Path, cuts: list[Cut]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "start_s",
                "end_s",
                "start_tc",
                "end_tc",
                "duration_s",
                "speaker",
                "reason",
            ]
        )
        for c in sorted(cuts, key=lambda x: (x.start_s, x.end_s)):
            w.writerow(
                [
                    f"{c.start_s:.3f}",
                    f"{c.end_s:.3f}",
                    timecode(c.start_s),
                    timecode(c.end_s),
                    f"{c.duration_s:.3f}",
                    c.speaker,
                    c.reason,
                ]
            )


def write_clips_csv(path: Path, clips) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "speaker",
                "orig_start_s",
                "orig_end_s",
                "orig_start_tc",
                "orig_end_tc",
                "new_start_s",
                "new_end_s",
                "new_start_tc",
                "new_end_tc",
            ]
        )
        for c in clips:
            w.writerow(
                [
                    c.speaker,
                    f"{c.src_start_s:.3f}",
                    f"{c.src_end_s:.3f}",
                    timecode(c.src_start_s),
                    timecode(c.src_end_s),
                    f"{c.dst_start_s:.3f}",
                    f"{c.dst_end_s:.3f}",
                    timecode(c.dst_start_s),
                    timecode(c.dst_end_s),
                ]
            )


def write_outputs(
    outdir: Path,
    sr: int,
    user_out: np.ndarray,
    asst_out: np.ndarray,
    combined: np.ndarray,
    user_removed: np.ndarray,
    asst_removed: np.ndarray,
    cuts: list[Cut],
    clips=None,
) -> dict[str, Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    paths = {
        "user": outdir / "USER.wav",
        "assistant": outdir / "ASSISTANT.wav",
        "combined": outdir / "COMBINED.wav",
        "user_removed": outdir / "USER_REMOVED.wav",
        "assistant_removed": outdir / "ASSISTANT_REMOVED.wav",
        "cuts": outdir / "cuts.csv",
        "clips": outdir / "clips.csv",
    }
    write_wav_24(paths["user"], user_out, sr)
    write_wav_24(paths["assistant"], asst_out, sr)
    write_wav_24(paths["combined"], combined, sr)
    write_wav_24(paths["user_removed"], user_removed, sr)
    write_wav_24(paths["assistant_removed"], asst_removed, sr)
    write_cuts_csv(paths["cuts"], cuts)
    write_clips_csv(paths["clips"], clips or [])
    return paths
