"""Command-line interface for the dual-track turn-taking editor."""

from __future__ import annotations

import argparse
from pathlib import Path

from ptsx.eval import evaluate_paths
from ptsx.pipeline import GateConfig, default_config, gate_files
from ptsx.turns import TurnParams
from ptsx.vad import VadParams


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ptsx",
        description="Gate two aligned speaker WAVs into exclusive turns. ptsx ingest is the SFT drop path.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    _add_gate(sub)
    _add_clean_cut(sub)
    _add_ingest(sub)
    _add_ingest_status(sub)
    _add_eval(sub)
    _add_script(sub)
    _add_app(sub)
    args = parser.parse_args(argv)
    if args.cmd == "gate":
        return _cmd_gate(args)
    if args.cmd == "clean-cut":
        return _cmd_clean_cut(args)
    if args.cmd == "ingest":
        return _cmd_ingest(args)
    if args.cmd == "ingest-status":
        return _cmd_ingest_status(args)
    if args.cmd == "script":
        return _cmd_script(args)
    if args.cmd == "app":
        from ptsx.app import main as app_main

        return app_main()
    return _cmd_eval(args)


def _add_io(p: argparse.ArgumentParser) -> None:
    p.add_argument("--user", required=True, type=Path, help="User (or speaker A) WAV")
    p.add_argument("--assistant", required=True, type=Path, help="Assistant (or speaker B) WAV")
    p.add_argument("--outdir", required=True, type=Path, help="Output directory (new files only)")


def _add_gate_options(p: argparse.ArgumentParser, *, conform: bool) -> None:
    p.add_argument("--fade-ms", type=float, default=10.0)
    p.add_argument("--strip-db", type=float, default=-28.0, help="Strip silence threshold (dBFS)")
    p.add_argument("--pad-ms", type=float, default=250.0, help="Pad kept clips on both sides")
    p.add_argument(
        "--user-transcript",
        type=Path,
        default=None,
        help="Optional Whisper JSON to label User turns in script.txt",
    )
    p.add_argument(
        "--assistant-transcript",
        type=Path,
        default=None,
        help="Optional Whisper JSON to label Assistant turns in script.txt",
    )
    p.add_argument("--min-gap", type=float, default=0.300)
    p.add_argument("--max-gap", type=float, default=0.800)
    p.add_argument("--target-gap", type=float, default=0.500)
    p.add_argument("--max-backchannel", type=float, default=0.55)
    p.add_argument(
        "--transcribe",
        action="store_true",
        help="Run Whisper and write labeled script.txt / script.csv",
    )
    p.add_argument(
        "--whisper-model",
        default="base",
        help="Whisper model when --transcribe is set (default: base)",
    )
    p.add_argument(
        "--chunk-minutes",
        type=float,
        default=None,
        help="Optional render windows snapped to turn gaps (default: entire file)",
    )
    if conform:
        p.add_argument(
            "--peak-min-db",
            type=float,
            default=-6.0,
            help="Boost kept speech peaks below this (dBFS, default: -6)",
        )
        p.add_argument(
            "--peak-max-db",
            type=float,
            default=-3.0,
            help="Reduce kept speech peaks above this (dBFS, default: -3)",
        )
        p.add_argument(
            "--no-conform",
            action="store_true",
            help="Leave original clip levels (no -6/-3 dB window)",
        )


def _add_gate(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("gate", help="Standard cut: strip, exclusive turns, -6/-3 dB window")
    _add_io(p)
    _add_gate_options(p, conform=True)


def _add_clean_cut(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "clean-cut",
        help="Plugin cleanup then cut (no -6/-3 window). dxRevive / Mouth De-click / Fresh Air / loudness",
    )
    _add_io(p)
    _add_gate_options(p, conform=False)


def _add_app(sub: argparse._SubParsersAction) -> None:
    sub.add_parser("app", help="Open the Mac window: pick two WAVs and an output folder")


def _add_script(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("script", help="Write script.txt from clips.csv plus Whisper JSON")
    p.add_argument("--clips", required=True, type=Path)
    p.add_argument("--user-transcript", required=True, type=Path)
    p.add_argument("--assistant-transcript", required=True, type=Path)
    p.add_argument("--outdir", required=True, type=Path)


def _cmd_script(args: argparse.Namespace) -> int:
    from ptsx.pipeline import _load_transcript
    from ptsx.script import (
        overlay_transcript,
        turns_from_clips_csv,
        write_script_csv,
        write_script_txt,
    )

    turns = turns_from_clips_csv(args.clips)
    rows = overlay_transcript(
        turns,
        _load_transcript(args.user_transcript),
        _load_transcript(args.assistant_transcript),
    )
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    write_script_txt(outdir / "script.txt", rows)
    write_script_csv(outdir / "script.csv", rows)
    labeled = sum(1 for *_, text in rows if text.strip())
    print(f"Wrote {len(rows)} turns ({labeled} with text) to {outdir / 'script.txt'}")
    return 0


def _add_eval(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("eval", help="Compare gated stems to gold USER/ASSISTANT bounces")
    p.add_argument("--pred-user", required=True, type=Path)
    p.add_argument("--pred-assistant", required=True, type=Path)
    p.add_argument("--gold-user", required=True, type=Path)
    p.add_argument("--gold-assistant", required=True, type=Path)


def _config_from_gate_args(args: argparse.Namespace, *, conform: bool) -> GateConfig:
    cfg = default_config()
    cfg.fade_ms = args.fade_ms
    cfg.chunk_minutes = args.chunk_minutes
    cfg.strip_db = args.strip_db
    cfg.pad_s = args.pad_ms / 1000.0
    cfg.user_transcript = args.user_transcript
    cfg.assistant_transcript = args.assistant_transcript
    cfg.transcribe = args.transcribe
    cfg.whisper_model = args.whisper_model
    cfg.conform = conform and not getattr(args, "no_conform", False)
    if hasattr(args, "peak_min_db"):
        cfg.peak_min_db = args.peak_min_db
        cfg.peak_max_db = args.peak_max_db
    cfg.turns = TurnParams(
        max_backchannel_s=args.max_backchannel,
        min_gap_s=args.min_gap,
        max_gap_s=args.max_gap,
        target_gap_s=args.target_gap,
    )
    cfg.vad = VadParams()
    return cfg


def _print_paths(outdir: Path, paths: dict[str, Path]) -> None:
    print(f"Wrote {len(paths)} files to {outdir.resolve()}")
    for key in (
        "user_clean",
        "assistant_clean",
        "user",
        "assistant",
        "combined",
        "user_removed",
        "assistant_removed",
        "cuts",
        "clips",
        "script",
        "script_csv",
        "editor",
        "summary",
    ):
        if key in paths:
            print(f"  {key}: {paths[key]}")


def _add_ingest_status(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "ingest-status",
        help="Print the dispatcher chat message from ingest_summary.json (no audio render)",
    )
    p.add_argument(
        "--outdir",
        required=True,
        type=Path,
        help="Package folder that already contains ingest_summary.json",
    )


def _cmd_ingest_status(args: argparse.Namespace) -> int:
    from ptsx.ingest import format_dispatcher_message, load_ingest_summary

    summary = load_ingest_summary(args.outdir)
    print(format_dispatcher_message(summary))
    return 0


def _add_ingest(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "ingest",
        help="SFT drop: find USER/ASSISTANT WAVs, clean-cut if plugins else gate, write EDITOR.txt",
    )
    p.add_argument(
        "--indir",
        type=Path,
        default=None,
        help="Folder with TASKID_USER.wav and TASKID_ASSISTANT.wav (default: ./drop)",
    )
    p.add_argument("--user", type=Path, default=None)
    p.add_argument("--assistant", type=Path, default=None)
    p.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Output folder (default: <indir>/out)",
    )
    p.add_argument(
        "--already-enhanced",
        action="store_true",
        help="Files already went through Adobe Enhance: gate only, no -6/-3 window",
    )
    p.add_argument(
        "--no-transcribe",
        action="store_true",
        help="Skip Whisper (script.txt will have times only)",
    )
    p.add_argument("--whisper-model", default="base")


def _cmd_ingest(args: argparse.Namespace) -> int:
    from ptsx.ingest import (
        format_dispatcher_message,
        ingest_dir,
        ingest_pair,
        resolve_ingest_indir,
    )

    kwargs = dict(
        already_enhanced=args.already_enhanced,
        transcribe=not args.no_transcribe,
        whisper_model=args.whisper_model,
    )
    indir = resolve_ingest_indir(args.indir, user=args.user)
    if indir is not None:
        paths = ingest_dir(indir, args.outdir, **kwargs)
        dest = (args.outdir or (indir / "out")).resolve()
    else:
        if args.user is None or args.assistant is None or args.outdir is None:
            raise SystemExit(
                "ptsx ingest needs files in ./drop, or --indir, "
                "or --user and --assistant and --outdir"
            )
        dest = args.outdir
        paths = ingest_pair(args.user, args.assistant, dest, **kwargs)
    _print_paths(dest, paths)
    summary_path = paths.get("summary")
    if summary_path and summary_path.is_file():
        import json

        print()
        print(format_dispatcher_message(json.loads(summary_path.read_text())))
    return 0


def _cmd_gate(args: argparse.Namespace) -> int:
    cfg = _config_from_gate_args(args, conform=True)
    paths = gate_files(args.user, args.assistant, args.outdir, cfg)
    _print_paths(args.outdir, paths)
    return 0


def _cmd_clean_cut(args: argparse.Namespace) -> int:
    from ptsx.pipeline import clean_cut_files

    cfg = _config_from_gate_args(args, conform=False)
    paths = clean_cut_files(args.user, args.assistant, args.outdir, cfg)
    _print_paths(args.outdir, paths)
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    report = evaluate_paths(
        args.pred_user, args.pred_assistant, args.gold_user, args.gold_assistant
    )
    print(report.as_text())
    return 1 if report.overlap_samples else 0


if __name__ == "__main__":
    raise SystemExit(main())
