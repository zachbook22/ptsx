"""Optional Whisper transcription for script.txt labels."""

from __future__ import annotations

import json
from pathlib import Path

from ptsx.audio import load_mono, resample_to

WHISPER_SR = 16000


def whisper_available() -> bool:
    try:
        import whisper  # noqa: F401

        return True
    except ImportError:
        return False


def transcribe_wav(
    path: Path,
    model,
) -> dict:
    """Transcribe a mono WAV. Uses ptsx's loader so Pro Tools BWF files work."""
    audio, sr = load_mono(path)
    audio16 = resample_to(audio, sr, WHISPER_SR)
    result = model.transcribe(
        audio16,
        language="en",
        condition_on_previous_text=False,
        fp16=False,
        verbose=False,
    )
    segs = [
        {"start": float(s["start"]), "end": float(s["end"]), "text": str(s["text"]).strip()}
        for s in result.get("segments", [])
    ]
    return {"text": result.get("text", ""), "segments": segs}


def transcribe_pair(
    user_path: Path,
    asst_path: Path,
    outdir: Path,
    model_name: str = "base",
) -> tuple[Path, Path]:
    if not whisper_available():
        raise SystemExit(
            "Whisper is not installed. From the ptsx folder run:\n"
            "  ./install.sh\n"
            "or: pip install openai-whisper"
        )
    import whisper

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"Loading Whisper '{model_name}' (first time downloads the model)...")
    model = whisper.load_model(model_name, device="cpu")
    user_json = outdir / "user.json"
    asst_json = outdir / "assistant.json"
    print(f"Transcribing user: {user_path}")
    user_json.write_text(json.dumps(transcribe_wav(user_path, model), indent=2))
    print(f"Transcribing assistant: {asst_path}")
    asst_json.write_text(json.dumps(transcribe_wav(asst_path, model), indent=2))
    return user_json, asst_json
