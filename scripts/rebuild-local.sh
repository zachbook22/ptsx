#!/usr/bin/env bash
# Recreate ptsx on this machine: venv, Whisper, PTSX.app (Mac), and drop/ inbox.
# Run from the repo, or copy-paste:
#   cd "$HOME/Pro Tools SX" && ./scripts/rebuild-local.sh
set -euo pipefail
cd "$(dirname "$0")/.."

INGEST_BRANCH="${PTSX_INGEST_BRANCH:-cursor/cloud-agent-1788978654702-74w5g}"

if [[ ! -f drop/README.md ]]; then
  echo "drop/ is not in this checkout. Fetching ${INGEST_BRANCH}..."
  git fetch origin "${INGEST_BRANCH}"
  if [[ -n "$(git status --porcelain)" ]]; then
    echo "Working tree has local changes. Stash or commit them, then:"
    echo "  git checkout ${INGEST_BRANCH}"
    echo "  ./scripts/rebuild-local.sh"
    exit 1
  fi
  git checkout "${INGEST_BRANCH}"
fi

mkdir -p drop
chmod +x install.sh scripts/ingest-drop.sh scripts/rebuild-local.sh

echo "Rebuilding ptsx in $(pwd)"
./install.sh

ROOT="$(pwd)"
echo
echo "Local rebuild finished."
echo "  repo: ${ROOT}"
echo "  drop: ${ROOT}/drop"
echo
echo "Copy hour WAVs onto disk (not into chat):"
echo "  ${ROOT}/drop/TASKID_USER.wav"
echo "  ${ROOT}/drop/TASKID_ASSISTANT.wav"
echo
echo "Then:"
echo "  source \"${ROOT}/.venv/bin/activate\""
echo "  ptsx ingest"
echo "  # or: \"${ROOT}/scripts/ingest-drop.sh\""
