# Local ingest drop

Copy hour-long bounces **here on disk**. Do not attach them in Cursor or GrokBot chat (~10 MB cap). Do not commit them; `*.wav` is gitignored.

```
drop/TASKID_USER.wav
drop/TASKID_ASSISTANT.wav
```

Mono, same sample rate.

```bash
source .venv/bin/activate
ptsx ingest
# or
./scripts/ingest-drop.sh
```

Package lands in `drop/out/`. On a licensed Mac with plugins this runs clean-cut; otherwise gate.
