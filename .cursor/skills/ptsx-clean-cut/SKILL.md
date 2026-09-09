---
name: ptsx-clean-cut
description: >-
  Run the ptsx plugin cleanup then cut job. Use when the user asks for
  clean-cut, plugin cleanup, dxRevive, Mouth De-click, Fresh Air, or
  loudness then arrange. Does not use PTSX.app.
disable-model-invocation: true
---

# ptsx clean-cut

Plugin cleanup on each raw, then the turn-taking cut. Never overwrite inputs. Do not open PTSX.app.

## Chain

1. Accentize dxRevive Mix 28%
2. RX 12 Mouth De-click 1.5%
3. Slate Fresh Air 5%
4. Loudness -21.9 LUFS integrated, -3.1 dBTP
5. `ptsx` gate **without** the -6/-3 peak window

## Steps

1. If `--user`, `--assistant`, or `--outdir` are missing, ask for both aligned mono WAVs and an output folder.
2. From the project root:

```bash
source .venv/bin/activate
ptsx clean-cut \
  --user /path/to/USER.wav \
  --assistant /path/to/ASSISTANT.wav \
  --outdir ./out \
  --transcribe
```

Omit `--transcribe` only if the user wants times without Whisper labels.

3. Abort if a plugin fails to load. Do not half-process.
4. Tell the user the output folder. `USER_CLEAN.wav` / `ASSISTANT_CLEAN.wav` are the cleaned raws; `USER.wav` / `ASSISTANT.wav` / `COMBINED.wav` are the arranged stems.
