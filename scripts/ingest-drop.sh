#!/usr/bin/env bash
# Run ptsx ingest on the repo-local drop/ inbox (hour WAVs stay on disk).
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi
exec ptsx ingest --indir drop "$@"
