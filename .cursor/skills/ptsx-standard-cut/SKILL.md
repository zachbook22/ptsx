---
name: ptsx-standard-cut
description: >-
  Run the ptsx standard cut only (no plugins). Use when the user asks for
  standard cut, gate only, or turn-taking without dxRevive / cleanup.
  Does not use PTSX.app.
disable-model-invocation: true
---

# ptsx standard cut

Strip silence, exclusive turns, 0.5 s gaps, and the -6/-3 dB speech window. No VST cleanup. Never overwrite inputs. Do not open PTSX.app.

## Steps

1. If `--user`, `--assistant`, or `--outdir` are missing, ask for both aligned mono WAVs and an output folder.
2. From the project root:

```bash
source .venv/bin/activate
ptsx gate \
  --user /path/to/USER.wav \
  --assistant /path/to/ASSISTANT.wav \
  --outdir ./out \
  --transcribe
```

Omit `--transcribe` only if the user wants times without Whisper labels.

3. Tell the user the output folder (`USER.wav`, `ASSISTANT.wav`, `COMBINED.wav`, `script.txt`).
