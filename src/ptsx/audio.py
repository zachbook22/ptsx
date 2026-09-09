"""Load and write mono WAV files without modifying the originals."""

from __future__ import annotations

from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def load_mono(path: Path) -> tuple[np.ndarray, int]:
    path = Path(path)
    if path.suffix.lower() == ".wav":
        return _load_riff_pcm(path)
    audio, sr = sf.read(str(path), always_2d=True, dtype="float32")
    if audio.shape[1] > 1:
        audio = audio.mean(axis=1)
    else:
        audio = audio[:, 0]
    return np.ascontiguousarray(audio, dtype=np.float32), int(sr)


def _load_riff_pcm(path: Path) -> tuple[np.ndarray, int]:
    """Read PCM WAV/BWF including Pro Tools bounces with extra chunks."""
    data = path.read_bytes()
    if len(data) < 12 or data[0:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("not a RIFF/WAVE file")
    pos = 12
    fmt = None
    pcm = None
    while pos + 8 <= len(data):
        cid = data[pos : pos + 4]
        size = int.from_bytes(data[pos + 4 : pos + 8], "little")
        start = pos + 8
        end = start + size
        if end > len(data):
            break
        if cid == b"fmt ":
            fmt = data[start:end]
        elif cid == b"data":
            pcm = data[start:end]
            break
        pos = end + (size % 2)
    if fmt is None or pcm is None:
        raise ValueError("missing fmt or data chunk")
    audio_format = int.from_bytes(fmt[0:2], "little")
    channels = int.from_bytes(fmt[2:4], "little")
    sr = int.from_bytes(fmt[4:8], "little")
    bits = int.from_bytes(fmt[14:16], "little")
    if audio_format not in (1, 0xFFFE):
        raise ValueError(f"unsupported WAV format {audio_format}")
    if bits == 24:
        usable = len(pcm) - (len(pcm) % 3)
        a = np.frombuffer(pcm[:usable], dtype=np.uint8).reshape(-1, 3)
        packed = (
            a[:, 0].astype(np.int32)
            | (a[:, 1].astype(np.int32) << 8)
            | (a[:, 2].astype(np.int32) << 16)
        )
        signed = np.where(packed & 0x800000, packed - 0x1000000, packed)
        audio = signed.astype(np.float32) / 8388608.0
    elif bits == 16:
        audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
    elif bits == 32:
        audio = np.frombuffer(pcm, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"unsupported bit depth {bits}")
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return np.ascontiguousarray(audio, dtype=np.float32), int(sr)


def match_lengths(*arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    n = min(len(a) for a in arrays)
    return tuple(a[:n] for a in arrays)


def resample_to(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return audio
    g = gcd(src_sr, dst_sr)
    up, down = dst_sr // g, src_sr // g
    out = resample_poly(audio, up, down)
    return np.ascontiguousarray(out, dtype=np.float32)


def write_wav_24(path: Path, audio: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(audio, -1.0, 1.0).astype(np.float32, copy=False)
    sf.write(str(path), clipped, sr, subtype="PCM_24")
