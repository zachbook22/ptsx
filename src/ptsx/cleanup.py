"""Light plugin cleanup, then loudness, before the editor. Never overwrites inputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from ptsx.audio import load_mono, write_wav_24

TARGET_LUFS = -21.9
TRUE_PEAK_DB = -3.1

DXREVIVE_VST3 = Path("/Library/Audio/Plug-Ins/VST3/Accentize-dxRevive.vst3")
MOUTH_DECLICK_VST3 = Path("/Library/Audio/Plug-Ins/VST3/RX 12 Mouth De-click.vst3")
FRESH_AIR_VST3 = Path("/Library/Audio/Plug-Ins/VST3/Slate Digital/Fresh Air.vst3")
CLEANUP_PLUGIN_PATHS = (DXREVIVE_VST3, MOUTH_DECLICK_VST3, FRESH_AIR_VST3)


def cleanup_plugins_present() -> bool:
    """True when the three clean-cut VST3s are on disk (does not load them)."""
    return all(p.exists() for p in CLEANUP_PLUGIN_PATHS)


def missing_cleanup_plugins() -> list[Path]:
    return [p for p in CLEANUP_PLUGIN_PATHS if not p.exists()]


@dataclass
class CleanupConfig:
    revive_mix: float = 28.0
    mouth_declick: float = 1.5
    fresh_air: float = 5.0
    target_lufs: float = TARGET_LUFS
    true_peak_db: float = TRUE_PEAK_DB


def default_cleanup() -> CleanupConfig:
    return CleanupConfig()


def db_to_amp(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def amp_to_db(amp: float) -> float:
    return float(20.0 * np.log10(max(float(amp), 1e-12)))


def integrated_lufs(audio: np.ndarray, sr: int) -> float:
    import pyloudnorm as pyln

    meter = pyln.Meter(sr)
    x = np.asarray(audio, dtype=np.float64).reshape(-1)
    return float(meter.integrated_loudness(x))


def true_peak_db(audio: np.ndarray, sr: int, oversample: int = 4) -> float:
    x = np.asarray(audio, dtype=np.float64).reshape(-1)
    up = resample_poly(x, oversample, 1)
    return amp_to_db(float(np.max(np.abs(up))))


def apply_loudness(
    audio: np.ndarray,
    sr: int,
    target_lufs: float = TARGET_LUFS,
    true_peak_db_lim: float = TRUE_PEAK_DB,
) -> np.ndarray:
    """Hit target integrated LUFS, then cap true peak. Silence-safe."""
    x = np.asarray(audio, dtype=np.float64).reshape(-1)
    peak = float(np.max(np.abs(x)))
    if peak < 1e-8:
        return np.ascontiguousarray(x, dtype=np.float32)
    try:
        loud = integrated_lufs(x, sr)
    except Exception:
        # All-quiet or too short for BS.1770; fall back to peak-only.
        loud = float("-inf")
    if np.isfinite(loud):
        x = x * db_to_amp(target_lufs - loud)
    tp = true_peak_db(x, sr)
    if tp > true_peak_db_lim:
        x = x * db_to_amp(true_peak_db_lim - tp)
    cap = db_to_amp(true_peak_db_lim)
    np.clip(x, -cap, cap, out=x)
    return np.ascontiguousarray(x, dtype=np.float32)


def _set_named(plugin, name: str, value: float) -> None:
    try:
        setattr(plugin, name, value)
    except Exception as exc:
        raise SystemExit(f"Could not set {type(plugin).__name__}.{name}={value}: {exc}") from exc


def load_cleanup_plugins(config: CleanupConfig | None = None) -> list:
    config = config or default_cleanup()
    try:
        from pedalboard import load_plugin
    except ImportError as exc:
        raise SystemExit(
            "pedalboard is required for clean-cut. From the ptsx folder:\n"
            "  pip install pedalboard pyloudnorm"
        ) from exc

    missing = missing_cleanup_plugins()
    if missing:
        raise SystemExit(
            "Cleanup plugins not found:\n" + "\n".join(f"  {p}" for p in missing)
        )

    try:
        revive = load_plugin(str(DXREVIVE_VST3))
        mouth = load_plugin(str(MOUTH_DECLICK_VST3))
        air = load_plugin(str(FRESH_AIR_VST3))
    except Exception as exc:
        raise SystemExit(
            "Could not load cleanup plugins (authorize them in a DAW once if needed).\n"
            f"{exc}"
        ) from exc

    _set_named(revive, "mix", config.revive_mix)
    _set_named(mouth, "sensitivity", config.mouth_declick)
    _set_named(air, "mid_air", config.fresh_air)
    _set_named(air, "high_air", config.fresh_air)
    print(
        f"Plugins: dxRevive mix={config.revive_mix:.1f}  "
        f"Mouth De-click sensitivity={config.mouth_declick:.2f}  "
        f"Fresh Air mid/high={config.fresh_air:.1f}"
    )
    return [revive, mouth, air]


def run_plugin_chain(
    audio: np.ndarray,
    sr: int,
    plugins: list,
) -> np.ndarray:
    from pedalboard import Pedalboard

    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    stereo = np.stack([x, x], axis=0)
    board = Pedalboard(plugins)
    try:
        out = board(stereo, sr)
    except Exception as exc:
        raise SystemExit(f"Cleanup plugin processing failed: {exc}") from exc
    if out.ndim == 2:
        out = out.mean(axis=0)
    return np.ascontiguousarray(out.astype(np.float32, copy=False))


def clean_audio(
    audio: np.ndarray,
    sr: int,
    config: CleanupConfig | None = None,
    plugins: list | None = None,
) -> np.ndarray:
    config = config or default_cleanup()
    if plugins is None:
        plugins = load_cleanup_plugins(config)
    print("Cleanup: dxRevive → Mouth De-click → Fresh Air")
    processed = run_plugin_chain(audio, sr, plugins)
    print(
        f"Cleanup: loudness {config.target_lufs:.1f} LUFS, "
        f"true peak {config.true_peak_db:.1f} dBTP"
    )
    return apply_loudness(
        processed, sr, config.target_lufs, config.true_peak_db
    )


def clean_pair(
    user: np.ndarray,
    asst: np.ndarray,
    sr: int,
    config: CleanupConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    config = config or default_cleanup()
    plugins = load_cleanup_plugins(config)
    print("Cleaning user…")
    user_c = clean_audio(user, sr, config, plugins)
    print("Cleaning assistant…")
    asst_c = clean_audio(asst, sr, config, plugins)
    return user_c, asst_c


def clean_files(
    user_path: Path,
    asst_path: Path,
    outdir: Path,
    config: CleanupConfig | None = None,
) -> tuple[Path, Path]:
    """Write USER_CLEAN.wav and ASSISTANT_CLEAN.wav. Never overwrite inputs."""
    user_path = Path(user_path).resolve()
    asst_path = Path(asst_path).resolve()
    outdir = Path(outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    user_out = outdir / "USER_CLEAN.wav"
    asst_out = outdir / "ASSISTANT_CLEAN.wav"
    for p in (user_out, asst_out):
        if p.resolve() in {user_path, asst_path}:
            raise SystemExit(f"Refusing to overwrite input file: {p}")
    user, sr_u = load_mono(user_path)
    asst, sr_a = load_mono(asst_path)
    if sr_u != sr_a:
        raise SystemExit(f"Sample rates differ: user {sr_u} Hz, assistant {sr_a} Hz")
    user_c, asst_c = clean_pair(user, asst, sr_u, config)
    write_wav_24(user_out, user_c, sr_u)
    write_wav_24(asst_out, asst_c, sr_u)
    print(f"Wrote {user_out}")
    print(f"Wrote {asst_out}")
    return user_out, asst_out
