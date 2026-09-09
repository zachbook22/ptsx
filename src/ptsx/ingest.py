"""SFT ingest: drop a USER/ASSISTANT pair, cut once, write the editor package.

Hour-long dual-track conversations are SFT audio. Run this on a licensed Mac
(VST3s + iLok). A dispatcher bot may kick the job and post stats; it must not
render audio or attach hour-long WAVs.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from ptsx.cleanup import cleanup_plugins_present, missing_cleanup_plugins
from ptsx.pipeline import clean_cut_files, default_config, gate_files
from ptsx.turns import TurnParams
from ptsx.vad import VadParams

WAV_SUFFIXES = {".wav", ".wave"}
_ROLE_SUFFIXES = (
    ("_user", "user"),
    ("_speaker_a", "user"),
    ("_assistant", "assistant"),
    ("_asst", "assistant"),
    ("_speaker_b", "assistant"),
)


def _role_from_name(path: Path) -> str | None:
    stem = path.stem.lower().replace("-", "_")
    if stem in {"user", "speaker_a", "speakera"}:
        return "user"
    if stem in {"assistant", "asst", "speaker_b", "speakerb"}:
        return "assistant"
    for suffix, role in _ROLE_SUFFIXES:
        if stem.endswith(suffix):
            return role
    return None


def task_id_from_path(path: Path) -> str | None:
    stem = path.stem
    lower = stem.lower().replace("-", "_")
    for suffix, _role in _ROLE_SUFFIXES:
        if lower.endswith(suffix):
            prefix = stem[: len(stem) - len(suffix)]
            return prefix or None
    return None


def task_id_from_pair(user: Path, assistant: Path) -> str:
    for path in (user, assistant):
        tid = task_id_from_path(path)
        if tid:
            return tid
    return Path(user).parent.name or "ingest"


def find_pair(indir: Path) -> tuple[Path, Path]:
    """Return (user, assistant) WAVs from a drop folder."""
    indir = Path(indir)
    if not indir.is_dir():
        raise SystemExit(f"Ingest folder is not a directory: {indir}")
    wavs = [
        p
        for p in indir.iterdir()
        if p.is_file() and p.suffix.lower() in WAV_SUFFIXES
    ]
    users = [p for p in wavs if _role_from_name(p) == "user"]
    assts = [p for p in wavs if _role_from_name(p) == "assistant"]
    if len(users) != 1 or len(assts) != 1:
        names = ", ".join(p.name for p in wavs) or "(none)"
        raise SystemExit(
            "Need exactly one USER and one ASSISTANT WAV in the ingest folder "
            f"(names like TASKID_USER.wav / TASKID_ASSISTANT.wav). Found: {names}"
        )
    return users[0].resolve(), assts[0].resolve()


def choose_mode(*, already_enhanced: bool, plugins_present: bool | None = None) -> str:
    """Return 'clean-cut' or 'gate'. Never Enhance and then clean-cut."""
    if already_enhanced:
        return "gate"
    present = cleanup_plugins_present() if plugins_present is None else plugins_present
    return "clean-cut" if present else "gate"


def _csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _count_clips(path: Path) -> int:
    return len(_csv_rows(path))


def _count_cuts(path: Path) -> dict[str, int]:
    reasons: Counter[str] = Counter()
    for row in _csv_rows(path):
        reason = (row.get("reason") or "unknown").strip() or "unknown"
        reasons[reason] += 1
    return dict(reasons)


def _overlap_bleed_count(cuts_by_reason: dict[str, int]) -> int:
    n = 0
    for reason, count in cuts_by_reason.items():
        key = reason.lower()
        if "bleed" in key or "overlap" in key:
            n += count
    return n


def write_editor_txt(
    outdir: Path,
    *,
    task_id: str,
    mode: str,
    already_enhanced: bool,
    plugins_present: bool,
    transcribe: bool = True,
    user: Path | None = None,
    assistant: Path | None = None,
) -> Path:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    chain = "gate --no-conform (already Enhanced; do not run VSTs on top)"
    if not already_enhanced:
        if mode == "clean-cut":
            chain = "clean-cut (dxRevive Mix 28% → Mouth De-click → Fresh Air → loudness, then gate with no -6/-3)"
        else:
            chain = "gate (cleanup plugins missing; licensed Mac needed for clean-cut)"
    lines = [
        "ptsx editor package (SFT ingest)",
        "",
        f"Task: {task_id}",
        f"Mode: {mode}",
        f"Chain: {chain}",
        f"Plugins present: {'yes' if plugins_present else 'no'}",
        f"Whisper: {'yes' if transcribe else 'no (times only)'}",
    ]
    if user is not None:
        lines.append(f"User raw: {user}")
    if assistant is not None:
        lines.append(f"Assistant raw: {assistant}")
    lines.extend(
        [
            "",
            "Working tracks (import these to edit):",
            "  USER.wav",
            "  ASSISTANT.wav",
            "",
            "Muted reference:",
            "  original bounces (the two raws above, never overwrite them)",
            "  USER_CLEAN.wav / ASSISTANT_CLEAN.wav (clean-cut only)",
            "",
            "Punch / restore:",
            "  USER_REMOVED.wav / ASSISTANT_REMOVED.wav (original timeline)",
            "  cuts.csv — overlap/bleed/backchannel drops",
            "  clips.csv — where each kept region moved",
            "  script.txt / script.csv — conversation list",
            "",
            "Pro Tools punch pass (leftover hangovers, not a full-day strip):",
            "  1. USER and ASSISTANT on top (working).",
            "  2. Raws muted below.",
            "  3. REMOVED to hear every strip; restore a false cut from REMOVED or raw.",
            "  4. Hangover = mute. Missing word = restore.",
            "",
            "Do not run Adobe Enhance and then clean-cut on the same files.",
            "Do not overwrite the raw bounces.",
            "Hour-long WAVs stay on the task share — not git, not chat attachments.",
            "",
        ]
    )
    path = outdir / "EDITOR.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def format_dispatcher_message(summary: dict) -> str:
    """Short chat text a dispatcher can post when ingest finishes.

    GrokBot (or any status bot) posts this. It must not upload WAVs.
    """
    dispatcher = summary.get("dispatcher") or {}
    task_id = summary.get("task_id") or dispatcher.get("task_id") or "unknown"
    mode = summary.get("mode") or dispatcher.get("mode") or "gate"
    outdir = (
        dispatcher.get("package_path")
        or summary.get("outdir")
        or dispatcher.get("outdir")
        or ""
    )
    clip_count = dispatcher.get("clip_count", summary.get("clips_total", 0))
    cut_count = dispatcher.get("cut_count", summary.get("cuts_total", 0))
    bleed = dispatcher.get(
        "overlap_bleed_count",
        _overlap_bleed_count(summary.get("cuts_by_reason") or {}),
    )
    cuts = dispatcher.get("cuts_by_reason") or summary.get("cuts_by_reason") or {}
    cut_bits = ", ".join(f"{k} {v}" for k, v in sorted(cuts.items())) or "none"
    return (
        f"ptsx ingest ready: {task_id} ({mode})\n"
        f"package: {outdir}\n"
        f"{clip_count} clips, {cut_count} cuts "
        f"(overlap/bleed {bleed}; {cut_bits})\n"
        "Editors: USER.wav + ASSISTANT.wav working; REMOVED to restore; "
        "Hangover = mute. Missing word = restore.\n"
        "GrokBot: do not attach hour-long WAVs."
    )


def write_ingest_summary(
    outdir: Path,
    *,
    task_id: str,
    mode: str,
    already_enhanced: bool,
    plugins_present: bool,
    transcribe: bool = True,
    user_in: Path | None = None,
    assistant_in: Path | None = None,
    extra: dict | None = None,
    paths: dict[str, Path] | None = None,
) -> Path:
    """Machine-readable status for a dispatcher (GrokBot posts this, not the WAVs)."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    cuts_by_reason = _count_cuts(outdir / "cuts.csv")
    clip_count = _count_clips(outdir / "clips.csv")
    cut_count = sum(cuts_by_reason.values())
    overlap_bleed = _overlap_bleed_count(cuts_by_reason)
    files = {key: str(p) for key, p in sorted((paths or {}).items())}
    payload: dict = {
        "status": "ready",
        "task_id": task_id,
        "mode": mode,
        "already_enhanced": already_enhanced,
        "plugins_present": plugins_present,
        "transcribe": transcribe,
        "outdir": str(outdir.resolve()),
        "user_in": str(user_in) if user_in is not None else None,
        "assistant_in": str(assistant_in) if assistant_in is not None else None,
        "clips_total": clip_count,
        "cuts_total": cut_count,
        "cuts_by_reason": cuts_by_reason,
        "files": files,
        "dispatcher": {
            "task_id": task_id,
            "mode": mode,
            "package_path": str(outdir.resolve()),
            "clip_count": clip_count,
            "cut_count": cut_count,
            "overlap_bleed_count": overlap_bleed,
            "cuts_by_reason": cuts_by_reason,
        },
        "grokbot": {
            "render_audio": False,
            "attach_wavs": False,
            "post_from": "ingest_summary.json",
            "kick_command": f"ptsx ingest --indir <share>/{task_id}",
        },
    }
    if extra:
        payload.update(extra)
        payload["dispatcher"].update(
            {k: v for k, v in extra.items() if k in {"whisper_model"}}
        )
    payload["dispatcher"]["message"] = format_dispatcher_message(payload)
    path = outdir / "ingest_summary.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def load_ingest_summary(path: Path) -> dict:
    path = Path(path)
    if path.is_dir():
        path = path / "ingest_summary.json"
    if not path.is_file():
        raise SystemExit(f"No ingest_summary.json at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def ingest_pair(
    user: Path,
    assistant: Path,
    outdir: Path,
    *,
    already_enhanced: bool = False,
    transcribe: bool = True,
    whisper_model: str = "base",
    plugins_present: bool | None = None,
) -> dict[str, Path]:
    user = Path(user).resolve()
    assistant = Path(assistant).resolve()
    outdir = Path(outdir).resolve()
    if user == assistant:
        raise SystemExit("User and assistant files must be different.")
    outdir.mkdir(parents=True, exist_ok=True)
    present = cleanup_plugins_present() if plugins_present is None else plugins_present
    mode = choose_mode(already_enhanced=already_enhanced, plugins_present=present)
    task_id = task_id_from_pair(user, assistant)
    cfg = default_config()
    cfg.transcribe = transcribe
    cfg.whisper_model = whisper_model
    cfg.turns = TurnParams()
    cfg.vad = VadParams()
    if mode == "clean-cut":
        print("Ingest: clean-cut (plugins present; skip Adobe Enhance)")
        paths = clean_cut_files(user, assistant, outdir, cfg)
    else:
        cfg.conform = not already_enhanced
        if already_enhanced:
            print("Ingest: gate --no-conform (already Enhanced; no VSTs on top)")
        else:
            missing = ", ".join(str(p) for p in missing_cleanup_plugins()) or "cleanup VSTs"
            print(f"Ingest: gate (plugins missing: {missing})")
        paths = gate_files(user, assistant, outdir, cfg)
    paths["editor"] = write_editor_txt(
        outdir,
        task_id=task_id,
        mode=mode,
        already_enhanced=already_enhanced,
        plugins_present=present,
        transcribe=transcribe,
        user=user,
        assistant=assistant,
    )
    paths["summary"] = write_ingest_summary(
        outdir,
        task_id=task_id,
        mode=mode,
        already_enhanced=already_enhanced,
        plugins_present=present,
        transcribe=transcribe,
        user_in=user,
        assistant_in=assistant,
        extra={"whisper_model": whisper_model},
        paths=paths,
    )
    return paths


def ingest_dir(
    indir: Path,
    outdir: Path | None = None,
    **kwargs,
) -> dict[str, Path]:
    user, assistant = find_pair(indir)
    dest = Path(outdir) if outdir is not None else Path(indir) / "out"
    return ingest_pair(user, assistant, dest, **kwargs)
