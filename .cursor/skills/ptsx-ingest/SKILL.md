---
name: ptsx-ingest
description: >-
  Run ptsx ingest on a licensed Mac. Picks clean-cut vs gate, writes the
  editing-task package and EDITOR.txt. Use when the user drops dual-track
  bounces or asks to process a task for SFT audio editing.
---

# ptsx ingest

Hour-long dual-track conversations are SFT audio. Run **ingest once** on the licensed Mac, then hand editors a punch package. Editors do not clone the repo or start from the raw hour.

## Drop naming

Copy the pair onto disk. Do not attach hour WAVs in chat (~10 MB cap).

```
drop/TASKID_USER.wav
drop/TASKID_ASSISTANT.wav
```

That `drop/` folder is the local inbox in this repo. `ptsx ingest` with no `--indir` uses it.

```bash
ptsx ingest
./scripts/ingest-drop.sh
```

Or any other folder: `ptsx ingest --indir /path/to/drop` writes to `drop/out/`.

## Which chain

- Plugins present (dxRevive, RX Mouth De-click, Fresh Air) and **not** already Enhanced → `clean-cut`.
- `--already-enhanced` → `gate` with `--no-conform` (no VSTs on top).
- Plugins missing → `gate` (standard -6/-3 window).
- Never Enhance and then clean-cut.

Adobe Enhance is a heavy AI restore. dxRevive Mix 28% alone is not a replacement. The full clean-cut chain can replace Enhance if an A/B still sounds like the talker.

## Editing-task package

Working `USER.wav` / `ASSISTANT.wav` on top in Pro Tools; raws muted; `USER_REMOVED.wav` / `ASSISTANT_REMOVED.wav` to restore. Hangover = mute; missing word = restore. `EDITOR.txt` restates this.

## GrokBot / dispatcher

GrokBot does **not** render audio and must not attach hour WAVs. A later bot may SSH/queue `ptsx ingest` on this Mac and post `ingest_summary.json` (package path, clip counts, overlap/bleed drops). Re-print without rendering:

```bash
ptsx ingest-status --outdir /path/to/TASKID/out
```

## Run

```bash
ptsx ingest --indir /path/to/TASKID
# already Enhanced:
ptsx ingest --indir /path/to/TASKID --already-enhanced
```
