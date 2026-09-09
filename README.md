# ptsx — dual-track turn-taking editor

Takes two aligned mono speaker WAVs (User and Assistant), keeps each person’s turns, and writes digital silence on the other track. Unanswered “yeah” / “mmhmm” / noise on the other track is dropped. Turns are not cut short. Clips are then nudged so speaker changes sit about 0.5 s apart. Original files are never overwritten.

## For Cursor coworkers

1. Get invited to this **private** GitHub repo, then clone it in Cursor and open the folder.
2. In a terminal at the project root, run the installer (Python 3.10+ and ffmpeg are installed if missing):

**Mac / Linux**

```bash
chmod +x install.sh
./install.sh
```

**Windows**

```bat
install.bat
```

On a Mac, Homebrew may ask for your password the first time. On Windows, App Installer / winget is used.

3. In Cursor chat, trigger **ptsx-standard-cut** or **ptsx-clean-cut** (attach the skill or type its name). They do not run on their own.

**Later updates:** `git pull` (or Cursor Source Control). Re-run the installer only if Python dependencies in `pyproject.toml` changed.

### Clean-cut plugins (not in this repo)

Standard cut needs no plugins. Clean-cut needs these licensed VST3s on the machine (typical Mac paths):

- `/Library/Audio/Plug-Ins/VST3/Accentize-dxRevive.vst3`
- `/Library/Audio/Plug-Ins/VST3/RX 12 Mouth De-click.vst3`
- `/Library/Audio/Plug-Ins/VST3/Slate Digital/Fresh Air.vst3`

### What the installer sets up

`.venv`, ptsx, NumPy, SciPy, onnxruntime, Whisper (`base` model), and on a Mac **PTSX.app**. Python, ffmpeg, the venv, and plugin binaries are not stored in git.

Optional Mac GUI: double-click **PTSX.app**, or `source .venv/bin/activate` then `ptsx app`. The two Cursor jobs use the CLI, not the app.

### Manual install

If you already have Python 3.10+ and ffmpeg on PATH:

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e .
```

## Run

Two jobs from Cursor or Terminal (not the Mac app for this):

**Standard cut** (no plugins; -6/-3 dB speech window):

```bash
source .venv/bin/activate
ptsx gate \
  --user /path/to/user.wav \
  --assistant /path/to/assistant.wav \
  --outdir ./out \
  --transcribe
```

**Clean-cut** (dxRevive 28% → RX 12 Mouth De-click 1.5% → Fresh Air 5% → -21.9 LUFS / -3.1 dBTP, then cut **without** the -6/-3 window):

```bash
source .venv/bin/activate
ptsx clean-cut \
  --user /path/to/user.wav \
  --assistant /path/to/assistant.wav \
  --outdir ./out \
  --transcribe
```

`--transcribe` labels `script.txt` with Whisper (the `base` model is downloaded during install). Omit it for a faster pass with times only.

Inputs are two **mono WAVs**, one per speaker. Clean-cut will time-align them if they were not started together. Standard cut expects matching length unless you align first.

Writes:

| File | What it is |
| --- | --- |
| `USER_CLEAN.wav` / `ASSISTANT_CLEAN.wav` | Plugin + loudness pass (`clean-cut` only) |
| `USER.wav` | User clips only, 250 ms pad, 10 ms fades on the pad |
| `ASSISTANT.wav` | Assistant clips only |
| `COMBINED.wav` | Sum of the two gated tracks (never overlapping) |
| `USER_REMOVED.wav` | Original timeline; audio that was stripped or not kept |
| `ASSISTANT_REMOVED.wav` | Same for Assistant |
| `cuts.csv` | Overlaps and short clips that were dropped |
| `clips.csv` | Each kept clip’s original time and new (nudged) time |
| `script.txt` | Turn-by-turn conversation (original timeline) |
| `script.csv` | Same script as a spreadsheet |
| `user.json` / `assistant.json` | Whisper output (when `--transcribe` is used) |

Inputs must be different files from anything in `--outdir`.

Useful flags: `--strip-db -28`, `--pad-ms 250`, `--fade-ms 10`, `--min-gap 0.3`, `--max-gap 0.8`, `--target-gap 0.5`, `--max-backchannel 0.55`, `--whisper-model base`, `--peak-min-db -6`, `--peak-max-db -3`, `--no-conform`.

If you already have Whisper JSON:

```bash
ptsx script \
  --clips ./out/clips.csv \
  --user-transcript ./out/user.json \
  --assistant-transcript ./out/assistant.json \
  --outdir ./out
```

## Human pass in Pro Tools

1. Import `USER.wav` and `ASSISTANT.wav` as the working tracks. These may be shorter than the raws because clips were nudged together.
2. Keep the original raw bounces on tracks below, muted.
3. Import `USER_REMOVED.wav` and `ASSISTANT_REMOVED.wav` (original timeline) to hear every strip/mute against the raws.
4. Use `script.txt` / `script.csv` as the conversation list, `clips.csv` for where each region moved, and `cuts.csv` as a punch list.

A leftover “yeah” is a quick mute; a false cut is a restore from REMOVED or raw.

## What it does

- **Standard cut (`ptsx gate`):** speech peaks below -6 dBFS are boosted to -6; peaks above -3 dBFS are limited to -3. Silence is not raised.
- **Clean-cut (`ptsx clean-cut`):** dxRevive 28%, RX 12 Mouth De-click 1.5%, Fresh Air 5%, then -21.9 LUFS / -3.1 dBTP; no -6/-3 window after that.
- Strip silence per track at -28 dBFS with 250 ms pad; quiet word tails are held so endings are not chopped.
- Build the conversation from unique-speaker turns. Nested yeahs/bleed are dropped; turns are not cut short.
- Nudge remaining clips so speaker-change gaps sit 0.300–0.800 s apart. Source clip lengths stay intact.
- 10 ms fade in/out on the pad, never on the dialogue itself.

Hour-long files run in one pass. Optional `--chunk-minutes 15` only splits render windows at turn gaps to limit memory.

## Tests (developers)

```bash
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

The bundled Silero VAD ONNX model (if present) is from [snakers4/silero-vad](https://github.com/snakers4/silero-vad) (MIT). Whisper is from OpenAI (`openai-whisper`).
