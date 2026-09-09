"""Compare gated stems to gold Pro Tools bounces."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ptsx.audio import load_mono, match_lengths


@dataclass
class EvalReport:
    user_iou: float
    asst_iou: float
    overlap_samples: int
    overlap_seconds: float
    user_missed_s: float
    user_leaked_s: float
    asst_missed_s: float
    asst_leaked_s: float
    sr: int

    def as_text(self) -> str:
        lines = [
            f"sample rate: {self.sr}",
            f"user IoU: {self.user_iou:.3f}",
            f"assistant IoU: {self.asst_iou:.3f}",
            f"overlapping non-zero samples: {self.overlap_samples} ({self.overlap_seconds:.4f}s)",
            f"user missed speech: {self.user_missed_s:.2f}s  leaked: {self.user_leaked_s:.2f}s",
            f"assistant missed speech: {self.asst_missed_s:.2f}s  leaked: {self.asst_leaked_s:.2f}s",
        ]
        if self.overlap_samples:
            lines.append("FAIL: gated tracks overlap")
        else:
            lines.append("PASS: no overlapping non-zero samples")
        return "\n".join(lines)


def activity_mask(audio: np.ndarray, sr: int, hop_s: float = 0.01, eps: float = 1e-5) -> np.ndarray:
    hop = max(1, int(round(sr * hop_s)))
    n = len(audio) // hop
    if n == 0:
        return np.zeros(0, dtype=bool)
    frames = audio[: n * hop].reshape(n, hop)
    return np.max(np.abs(frames), axis=1) > eps


def iou(a: np.ndarray, b: np.ndarray) -> float:
    both = np.sum(a & b)
    union = np.sum(a | b)
    if union == 0:
        return 1.0
    return float(both / union)


def evaluate_stems(
    pred_user: np.ndarray,
    pred_asst: np.ndarray,
    gold_user: np.ndarray,
    gold_asst: np.ndarray,
    sr: int,
    hop_s: float = 0.01,
) -> EvalReport:
    pred_user, pred_asst, gold_user, gold_asst = match_lengths(
        pred_user, pred_asst, gold_user, gold_asst
    )
    overlap = int(np.sum((np.abs(pred_user) > 0.0) & (np.abs(pred_asst) > 0.0)))
    pu = activity_mask(pred_user, sr, hop_s)
    pa = activity_mask(pred_asst, sr, hop_s)
    gu = activity_mask(gold_user, sr, hop_s)
    ga = activity_mask(gold_asst, sr, hop_s)
    n = min(len(pu), len(pa), len(gu), len(ga))
    pu, pa, gu, ga = pu[:n], pa[:n], gu[:n], ga[:n]
    return EvalReport(
        user_iou=iou(pu, gu),
        asst_iou=iou(pa, ga),
        overlap_samples=overlap,
        overlap_seconds=overlap / sr,
        user_missed_s=float(np.sum(gu & ~pu) * hop_s),
        user_leaked_s=float(np.sum(pu & ~gu) * hop_s),
        asst_missed_s=float(np.sum(ga & ~pa) * hop_s),
        asst_leaked_s=float(np.sum(pa & ~ga) * hop_s),
        sr=sr,
    )


def evaluate_paths(pred_user, pred_asst, gold_user, gold_asst) -> EvalReport:
    pu, sr = load_mono(pred_user)
    pa, sr_a = load_mono(pred_asst)
    gu, sr_gu = load_mono(gold_user)
    ga, sr_ga = load_mono(gold_asst)
    if len({sr, sr_a, sr_gu, sr_ga}) != 1:
        raise SystemExit("All eval files must share a sample rate")
    return evaluate_stems(pu, pa, gu, ga, sr)
