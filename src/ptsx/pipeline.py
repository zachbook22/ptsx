"""End-to-end gate: read the conversation → keep full turns → strip → nudge."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ptsx.audio import load_mono, match_lengths
from ptsx.gain import (
    DEFAULT_MAX_DB,
    DEFAULT_MIN_DB,
    conform_level,
    needs_detection_boost,
)
from ptsx.layout import place_time_clips, removed_stems, render_placed
from ptsx.render import write_outputs
from ptsx.script import (
    assign_script_turns,
    overlay_transcript,
    write_script_csv,
    write_script_txt,
)
from ptsx.strip import (
    auto_hold_db,
    clips_from_mask,
    frame_peak,
    merge_near_clips,
    strip_silence,
)
from ptsx.turns import Cut, TurnParams, chunk_sample_bounds, merge_cuts
from ptsx.vad import VadParams


STRIP_HOP_S = 0.010


@dataclass
class GateConfig:
    vad: VadParams
    turns: TurnParams
    fade_ms: float = 10.0
    chunk_minutes: float | None = None
    strip_db: float = -28.0
    pad_s: float = 0.250
    unique_db: float = 6.0
    user_transcript: Path | None = None
    assistant_transcript: Path | None = None
    transcribe: bool = False
    whisper_model: str = "base"
    conform: bool = True
    peak_min_db: float = DEFAULT_MIN_DB
    peak_max_db: float = DEFAULT_MAX_DB


def default_config() -> GateConfig:
    return GateConfig(vad=VadParams(), turns=TurnParams())


def gate_arrays(
    user: np.ndarray,
    asst: np.ndarray,
    sr: int,
    config: GateConfig | None = None,
    vad=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list, list]:
    del vad  # Strip silence replaces VAD for clip finding.
    config = config or default_config()
    user, asst = match_lengths(user, asst)
    user_src, asst_src = user, asst
    hop_s = STRIP_HOP_S
    user_render, asst_render = user, asst
    user_det, asst_det = user, asst
    if config.conform:
        print(
            f"Level: boost speech below {config.peak_min_db:.0f} dBFS, "
            f"limit above {config.peak_max_db:.0f} dBFS"
        )
        user_render = conform_level(
            user,
            sr,
            min_db=config.peak_min_db,
            max_db=config.peak_max_db,
            strip_db=config.strip_db,
        )
        asst_render = conform_level(
            asst,
            sr,
            min_db=config.peak_min_db,
            max_db=config.peak_max_db,
            strip_db=config.strip_db,
        )
        # Only raise a quiet mic for strip/uniqueness. Boosting a hot mic
        # first glues its room tone into one endless floor.
        user_det = (
            user_render
            if needs_detection_boost(user_src, sr, hop_s, config.strip_db)
            else user_src
        )
        asst_det = (
            asst_render
            if needs_detection_boost(asst_src, sr, hop_s, config.strip_db)
            else asst_src
        )
    n = len(user_src)

    strip_u = strip_silence(
        user_det,
        sr,
        config.strip_db,
        config.pad_s,
        hop_s,
        hold_db=auto_hold_db(user_det, sr, hop_s, config.strip_db),
    )
    strip_a = strip_silence(
        asst_det,
        sr,
        config.strip_db,
        config.pad_s,
        hop_s,
        hold_db=auto_hold_db(asst_det, sr, hop_s, config.strip_db),
    )
    n_frames = min(len(strip_u), len(strip_a))
    strip_u, strip_a = strip_u[:n_frames], strip_a[:n_frames]
    peak_u = frame_peak(user_render, sr, hop_s)[:n_frames]
    peak_a = frame_peak(asst_render, sr, hop_s)[:n_frames]

    src_clips = clips_from_mask(strip_u, hop_s, "user") + clips_from_mask(
        strip_a, hop_s, "assistant"
    )
    src_clips = merge_near_clips(src_clips, max_gap_s=0.50)
    kept, nested_dropped = assign_script_turns(
        src_clips,
        peak_u,
        peak_a,
        hop_s,
        unique_db=config.unique_db,
        max_backchannel_s=max(config.turns.max_backchannel_s, 0.70),
    )
    cuts = [Cut(s, e, sp, "overlap") for sp, s, e in nested_dropped]
    cuts = merge_cuts(cuts)
    dropped_s = [(c.start_s, c.end_s) for c in cuts]

    if config.chunk_minutes:
        from ptsx.strip import clips_to_mask

        n_frames = min(len(strip_u), len(strip_a))
        keep_u = clips_to_mask(kept, hop_s, n_frames, "user")
        keep_a = clips_to_mask(kept, hop_s, n_frames, "assistant")
        chunk_sample_bounds(keep_u, keep_a, hop_s, sr, n, config.chunk_minutes)

    placed = place_time_clips(kept, config.turns, dropped_s=dropped_s)
    user_out, asst_out, combined = render_placed(
        user_render, asst_render, placed, sr, config.fade_ms
    )
    user_removed, asst_removed = removed_stems(user_src, asst_src, placed, sr)
    return user_out, asst_out, combined, user_removed, asst_removed, cuts, placed


def gate_files(
    user_path: Path,
    asst_path: Path,
    outdir: Path,
    config: GateConfig | None = None,
) -> dict[str, Path]:
    user_path = user_path.resolve()
    asst_path = asst_path.resolve()
    outdir = outdir.resolve()
    if user_path == asst_path:
        raise SystemExit("User and assistant files must be different.")

    user, sr_u = load_mono(user_path)
    asst, sr_a = load_mono(asst_path)
    if sr_u != sr_a:
        raise SystemExit(f"Sample rates differ: user {sr_u} Hz, assistant {sr_a} Hz")

    planned = [
        outdir / "USER.wav",
        outdir / "ASSISTANT.wav",
        outdir / "COMBINED.wav",
        outdir / "USER_REMOVED.wav",
        outdir / "ASSISTANT_REMOVED.wav",
    ]
    for p in planned:
        if p.resolve() in {user_path, asst_path}:
            raise SystemExit(f"Refusing to overwrite input file: {p}")

    user_out, asst_out, combined, user_removed, asst_removed, cuts, clips = gate_arrays(
        user, asst, sr_u, config
    )
    paths = write_outputs(
        outdir,
        sr_u,
        user_out,
        asst_out,
        combined,
        user_removed,
        asst_removed,
        cuts,
        clips,
    )
    config = config or default_config()
    if config.transcribe and not (config.user_transcript and config.assistant_transcript):
        from ptsx.transcribe import transcribe_pair

        user_json, asst_json = transcribe_pair(
            user_path, asst_path, outdir, model_name=config.whisper_model
        )
        config.user_transcript = user_json
        config.assistant_transcript = asst_json
    turns = [(c.speaker, c.src_start_s, c.src_end_s) for c in clips]
    user_segs = _load_transcript(config.user_transcript)
    asst_segs = _load_transcript(config.assistant_transcript)
    rows = overlay_transcript(turns, user_segs, asst_segs)
    script_path = outdir / "script.txt"
    write_script_txt(script_path, rows)
    csv_path = outdir / "script.csv"
    write_script_csv(csv_path, rows)
    paths["script"] = script_path
    paths["script_csv"] = csv_path
    return paths


def clean_cut_files(
    user_path: Path,
    asst_path: Path,
    outdir: Path,
    config: GateConfig | None = None,
) -> dict[str, Path]:
    """Align if needed, plugin cleanup, then gate without the -6/-3 peak window."""
    from ptsx.align import align_exclusive
    from ptsx.audio import load_mono, write_wav_24
    from ptsx.cleanup import clean_pair

    config = config or default_config()
    config.conform = False
    user_path = Path(user_path).resolve()
    asst_path = Path(asst_path).resolve()
    outdir = Path(outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    user, sr_u = load_mono(user_path)
    asst, sr_a = load_mono(asst_path)
    if sr_u != sr_a:
        raise SystemExit(f"Sample rates differ: user {sr_u} Hz, assistant {sr_a} Hz")
    user, asst, lag_s = align_exclusive(user, asst, sr_u)
    print(f"Align: pad {'user' if lag_s < 0 else 'assistant' if lag_s > 0 else 'neither'} "
          f"start by {abs(lag_s):.2f}s so speech does not overlap")
    print("Cleaning aligned tracks…")
    user_c, asst_c = clean_pair(user, asst, sr_u)
    user_clean = outdir / "USER_CLEAN.wav"
    asst_clean = outdir / "ASSISTANT_CLEAN.wav"
    for p in (user_clean, asst_clean):
        if p.resolve() in {user_path, asst_path}:
            raise SystemExit(f"Refusing to overwrite input file: {p}")
    write_wav_24(user_clean, user_c, sr_u)
    write_wav_24(asst_clean, asst_c, sr_u)
    print(f"Wrote {user_clean}")
    print(f"Wrote {asst_clean}")
    paths = gate_files(user_clean, asst_clean, outdir, config)
    paths["user_clean"] = user_clean
    paths["assistant_clean"] = asst_clean
    return paths


def _load_transcript(path: Path | None) -> list[dict]:
    if path is None:
        return []
    p = Path(path)
    if not p.is_file():
        return []
    import json

    data = json.loads(p.read_text())
    return list(data.get("segments", []))
