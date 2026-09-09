"""Silero VAD (ONNX) plus hysteresis segmentation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort

VAD_SR = 16000
WINDOW = 512  # 32 ms at 16 kHz
CONTEXT = 64


def model_path() -> Path:
    return Path(__file__).resolve().parent / "models" / "silero_vad.onnx"


class SileroVAD:
    def __init__(self, path: Path | None = None) -> None:
        onnx = str(path or model_path())
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self.session = ort.InferenceSession(
            onnx, sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self.input_names = [i.name for i in self.session.get_inputs()]

    def probabilities(self, audio_16k: np.ndarray) -> np.ndarray:
        x = np.ascontiguousarray(audio_16k, dtype=np.float32)
        pad = (WINDOW - (len(x) % WINDOW)) % WINDOW
        if pad:
            x = np.pad(x, (0, pad))
        n_frames = len(x) // WINDOW
        probs = np.empty(n_frames, dtype=np.float32)
        state = np.zeros((2, 1, 128), dtype=np.float32)
        context = np.zeros((1, CONTEXT), dtype=np.float32)
        sr = np.array([VAD_SR], dtype=np.int64)

        for i in range(n_frames):
            chunk = x[i * WINDOW : (i + 1) * WINDOW].reshape(1, WINDOW)
            inp = np.concatenate([context, chunk], axis=1)
            feeds = {"input": inp, "state": state}
            if "sr" in self.input_names:
                feeds["sr"] = sr
            out = self.session.run(None, feeds)
            probs[i] = float(np.asarray(out[0]).reshape(-1)[0])
            state = out[1]
            context = inp[:, -CONTEXT:]
        return probs


@dataclass
class VadParams:
    threshold: float = 0.5
    neg_threshold: float = 0.35
    min_speech_ms: float = 90.0
    min_silence_ms: float = 120.0
    speech_pad_ms: float = 80.0
    energy_noise_percentile: float = 15.0
    energy_ratio: float = 4.0
    energy_min: float = 0.002


def probs_to_mask(probs: np.ndarray, hop_s: float, params: VadParams) -> np.ndarray:
    """Hysteresis VAD → boolean mask at the same frame rate as `probs`."""
    n = len(probs)
    speech = np.zeros(n, dtype=bool)
    triggered = False
    for i, p in enumerate(probs):
        if not triggered:
            if p >= params.threshold:
                triggered = True
                speech[i] = True
        else:
            if p >= params.neg_threshold:
                speech[i] = True
            else:
                triggered = False

    min_speech = max(1, int(round(params.min_speech_ms / 1000.0 / hop_s)))
    min_silence = max(1, int(round(params.min_silence_ms / 1000.0 / hop_s)))
    speech = _drop_short(speech, min_speech)
    speech = _fill_short_gaps(speech, min_silence)

    pad = int(round(params.speech_pad_ms / 1000.0 / hop_s))
    if pad > 0:
        speech = _dilate(speech, pad)
    return speech


def frame_pitch(
    audio_16k: np.ndarray, hop: int = WINDOW, sr: int = VAD_SR
) -> tuple[np.ndarray, np.ndarray]:
    """Return (f0_hz, periodicity) at one value per `hop` samples."""
    n = len(audio_16k) // hop
    f0 = np.zeros(n, dtype=np.float32)
    strength = np.zeros(n, dtype=np.float32)
    if n <= 0:
        return f0, strength
    frames = np.ascontiguousarray(audio_16k[: n * hop], dtype=np.float32).reshape(n, hop)
    lo = max(1, int(sr / 400))
    hi = min(hop - 1, int(sr / 70))
    for i in range(n):
        x = frames[i] - float(frames[i].mean())
        energy = float(np.sqrt(np.mean(x * x)))
        if energy < 0.003:
            continue
        ac = np.correlate(x, x, mode="full")[hop - 1 :]
        peak_i = int(np.argmax(ac[lo:hi]))
        denom = float(ac[0]) + 1e-12
        strength[i] = float(ac[lo + peak_i] / denom)
        f0[i] = float(sr / (lo + peak_i))
    return f0, strength


def frame_rms(audio: np.ndarray, sr: int, hop_s: float) -> np.ndarray:
    hop = max(1, int(round(sr * hop_s)))
    n = len(audio) // hop
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    frames = audio[: n * hop].reshape(n, hop)
    return np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1) + 1e-18).astype(
        np.float32
    )


def energy_mask(rms: np.ndarray, hop_s: float, params: VadParams) -> np.ndarray:
    if rms.size == 0:
        return np.zeros(0, dtype=bool)
    noise = float(np.percentile(rms, params.energy_noise_percentile))
    mask = rms > max(noise * params.energy_ratio, params.energy_min)
    min_speech = max(1, int(round(params.min_speech_ms / 1000.0 / hop_s)))
    min_silence = max(1, int(round(params.min_silence_ms / 1000.0 / hop_s)))
    mask = _drop_short(mask, min_speech)
    mask = _fill_short_gaps(mask, min_silence)
    return mask


def combined_activity(
    probs: np.ndarray, rms: np.ndarray, hop_s: float, params: VadParams
) -> np.ndarray:
    n = min(len(probs), len(rms))
    silero = probs_to_mask(probs[:n], hop_s, params)
    energy = energy_mask(rms[:n], hop_s, params)
    return silero | energy


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


def _fill_short_gaps(mask: np.ndarray, min_gap: int) -> np.ndarray:
    out = mask.copy()
    n = len(out)
    i = 0
    while i < n:
        if not out[i]:
            j = i
            while j < n and not out[j]:
                j += 1
            # only fill interior gaps
            if i > 0 and j < n and (j - i) < min_gap:
                out[i:j] = True
            i = j
        else:
            i += 1
    return out


def _dilate(mask: np.ndarray, pad: int) -> np.ndarray:
    if pad <= 0:
        return mask
    n = len(mask)
    out = mask.copy()
    idx = np.where(mask)[0]
    if idx.size == 0:
        return out
    for i in idx:
        out[max(0, i - pad) : min(n, i + pad + 1)] = True
    return out


def mask_to_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    segs: list[tuple[int, int]] = []
    n = len(mask)
    i = 0
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            segs.append((i, j))
            i = j
        else:
            i += 1
    return segs
