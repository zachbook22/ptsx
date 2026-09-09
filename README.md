# ptsx — dual-track turn-taking editor

Takes two aligned mono speaker WAVs (User and Assistant), keeps each person’s turns, and writes digital silence on the other track. Unanswered “yeah” / “mmhmm” / noise on the other track is dropped. Turns are not cut short. Clips are then nudged so speaker changes sit about 0.5 s apart. Original files are never overwritten.

Hour-long dual-track conversations used as SFT audio are ingested **once** on a licensed Mac (`ptsx ingest`). Editors then get a punch package. They do not clone this repo or start from the hour-long dual raws. Software lives in git; audio stays on the task share.

## SFT ingest

Recorders drop two mono WAVs onto a dedicated Mac with this private ptsx install (VST3s + iLok). Launchd or a folder action on that Mac is enough to go from drop to done without Cursor.

### What to drop in

- `TASKID_USER.wav` / `TASKID_ASSISTANT.wav`
- Mono, same sample rate. Clean-cut time-aligns if they were not started together.

Do **not** attach hour WAVs in Cursor or GrokBot chat (there is a ~10 MB cap). Copy them onto disk.

**Local inbox in this repo:** put the pair in `drop/` (gitignored, not indexed):

```
drop/TASKID_USER.wav
drop/TASKID_ASSISTANT.wav
```

```bash
source .venv/bin/activate
ptsx ingest
# same as:
./scripts/ingest-drop.sh
# already ran Adobe Enhance (legacy):
ptsx ingest --already-enhanced
```

Writes to `drop/out/`. Or point at any folder:

```bash
ptsx ingest --indir /path/to/TASKID
```

You can also pass `--user`, `--assistant`, and `--outdir`. Ingest transcribes by default; `--no-transcribe` skips Whisper.

### Rebuild on your Mac

This cloud workspace is not the Mac. Recreate the install next to the files you will drop:

```bash
cd "$HOME/Pro Tools SX"
chmod +x scripts/rebuild-local.sh
./scripts/rebuild-local.sh
```

That checks out the ingest branch if `drop/` is missing, rebuilds `.venv` / Whisper / `PTSX.app`, and prints the absolute `drop/` path. Then copy `TASKID_USER.wav` and `TASKID_ASSISTANT.wav` into that folder (Finder or `cp`), not into chat.

### Enhance vs clean-cut vs gate

Adobe Enhance is a heavy AI restore (noise, reverb, “studio mic” leveling). dxRevive at Mix 28% alone is **not** a replacement: clean-cut runs dxRevive as a light wet/dry mix, then Mouth De-click and Fresh Air, then -21.9 LUFS / -3.1 dBTP. That is production dialogue cleanup, not Enhance’s full rewrite of the voice.

The **whole** clean-cut chain can replace Enhance for this pipeline if a 5-minute A/B on the same raw still sounds like the talker. For SFT data that is usually the better default: Enhance often adds AI timbre and pumping. Clean-cut stays local, licensed, and repeatable.

**Never Enhance and then clean-cut** (double processing). Pick one:

| Drop | Command |
| --- | --- |
| Raw + licensed Mac (plugins present) | `ptsx ingest` → `clean-cut --transcribe` (skip Enhance). Raise `revive_mix` later only if rooms are worse than 28% can handle. |
| Already Enhanced (legacy) | `ptsx ingest --already-enhanced` → `gate --no-conform --transcribe` only. No VSTs on top. |
| Plugins missing | `ptsx ingest` → `gate` (standard -6/-3 window). Use a licensed Mac for clean-cut. |

Do a 5-minute A/B (Enhance vs clean-cut on the same raw) before you delete Enhance from the bounce SOP. If clean-cut wins or ties, drop Enhance from new tasks.

### What goes in the editing task

- Working: `USER.wav`, `ASSISTANT.wav`
- Muted reference: original bounces (and `*_CLEAN.wav` if clean-cut ran)
- Punch: `USER_REMOVED.wav`, `ASSISTANT_REMOVED.wav`
- List: `script.txt` / `script.csv`, `clips.csv`, `cuts.csv`
- Notes: `EDITOR.txt` (Pro Tools punch steps)
- Dispatcher: `ingest_summary.json` (clip/cut counts for a status bot)

### Pro Tools punch pass

Layout: working pair on top, raws muted, REMOVED to restore. Hangover = mute; missing word = restore. Isolated mics should be light. This is a leftover hangover pass, not a full-day strip.

### Time expectation

- Ingest unattended: gate a few minutes; clean-cut ~15–30+ min plus Whisper.
- Editor: leftover hangovers. Goal is well under a day, not zero listen.

### GrokBot (dispatcher only)

GrokBot must **not** render the hour of audio. Clean-cut needs local VST3s and iLok on a Mac; hour WAVs must not go through chat or git.

A later bot may:

1. Take `process TASKID` or a path on the ingest share (not an attached WAV).
2. SSH or queue `ptsx ingest --indir …` on the licensed Mac.
3. Post the package path plus punch-list stats from `ingest_summary.json` (clip counts, `cuts.csv` overlap/bleed drops). Re-print without rendering:

```bash
ptsx ingest-status --outdir /path/to/TASKID/out
```

Optional later: answer “what got cut around 12:55?” from `clips.csv` / `script.txt` already on disk.

GrokBot must not upload hour-long bounces, run dxRevive in the cloud, or replace the Pro Tools listen. Editors still punch hangovers; the bot only starts the job and reports.

Cursor skills stay on the ingest Mac for operators who prefer chat there. They are not the scaled editor path.

Out of scope: hour WAVs in GitHub or in GrokBot attachments; wrapping Adobe Enhance inside ptsx; every editor installing VSTs or Cursor.

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

3. In Cursor chat on the ingest Mac, trigger **ptsx-ingest** for an SFT drop (or **ptsx-standard-cut** / **ptsx-clean-cut** for a one-off). They do not run on their own.

**Later updates:** `git pull` (or Cursor Source Control). Re-run the installer only if Python dependencies in `pyproject.toml` changed.

### Clean-cut plugins (not in this repo)

Standard cut needs no plugins. Clean-cut needs these licensed VST3s on the machine (typical Mac paths):

- `/Library/Audio/Plug-Ins/VST3/Accentize-dxRevive.vst3`
- `/Library/Audio/Plug-Ins/VST3/RX 12 Mouth De-click.vst3`
- `/Library/Audio/Plug-Ins/VST3/Slate Digital/Fresh Air.vst3`

### What the installer sets up

`.venv`, ptsx, NumPy, SciPy, onnxruntime, Whisper (`base` model), and on a Mac **PTSX.app**. Python, ffmpeg, the venv, and plugin binaries are not stored in git.

Optional Mac GUI: double-click **PTSX.app**, or `source .venv/bin/activate` then `ptsx app`. Cursor skills on the ingest Mac use the CLI, not the app.

### Manual install

If you already have Python 3.10+ and ffmpeg on PATH:

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e .
```

## Run (operator one-offs)

SFT drops should use `ptsx ingest` above. These two commands are for a single pair when you already know the chain:

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
| `EDITOR.txt` | Punch-pass notes (`ptsx ingest`) |
| `ingest_summary.json` | Dispatcher stats for GrokBot (`ptsx ingest`) |

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
- Build the conversation from unique-speaker turns. Nested yeahs/bleed are dropped; turns are not cut short. A clip that is quieter than the other mic at the same time is treated as bleed. Short leftovers at a speaker change are dropped if they lose to the other mic on the overlapping frames (even when the leftover looks unique after that person stops). Quiet one-sided pickup before the other speaker’s first turn is dropped too.
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
