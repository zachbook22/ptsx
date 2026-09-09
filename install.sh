#!/usr/bin/env bash
# Install ptsx: Python 3.10+, ffmpeg, venv, Whisper base.
set -euo pipefail
cd "$(dirname "$0")"

is_python_ok() {
  local bin="$1"
  [[ -x "$bin" ]] || return 1
  "$bin" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null
}

find_python() {
  local c resolved prefix formula
  if command -v brew >/dev/null 2>&1; then
    for formula in python@3.13 python@3.12 python@3.11 python@3.10; do
      prefix="$(brew --prefix "$formula" 2>/dev/null || true)"
      if [[ -n "$prefix" ]]; then
        for c in "$prefix/bin/python3.13" "$prefix/bin/python3.12" "$prefix/bin/python3.11" "$prefix/bin/python3.10" "$prefix/bin/python3"; do
          if is_python_ok "$c"; then
            printf '%s\n' "$c"
            return 0
          fi
        done
      fi
    done
  fi
  for c in python3.13 python3.12 python3.11 python3.10 python3; do
    resolved="$(command -v "$c" 2>/dev/null || true)"
    if [[ -n "$resolved" ]] && is_python_ok "$resolved"; then
      printf '%s\n' "$resolved"
      return 0
    fi
  done
  return 1
}

ensure_homebrew() {
  if command -v brew >/dev/null 2>&1; then
    return 0
  fi
  if [[ "$(uname -s)" != Darwin ]]; then
    return 1
  fi
  echo "Homebrew is not installed. Installing Homebrew (may ask for your password)..."
  NONINTERACTIVE=1 /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  if [[ -x /opt/homebrew/bin/brew ]]; then
    eval "$(/opt/homebrew/bin/brew shellenv)"
  elif [[ -x /usr/local/bin/brew ]]; then
    eval "$(/usr/local/bin/brew shellenv)"
  fi
  command -v brew >/dev/null 2>&1
}

install_mac_python_ffmpeg() {
  ensure_homebrew || {
    echo "Could not install Homebrew. Install Python 3.10+ and ffmpeg, then re-run."
    echo "  https://brew.sh"
    exit 1
  }
  echo "Installing Python 3.12 and ffmpeg with Homebrew..."
  brew install python@3.12 ffmpeg
  local pyprefix
  pyprefix="$(brew --prefix python@3.12)"
  export PATH="$pyprefix/bin:$PATH"
  hash -r
}

PY=""
if PY="$(find_python)"; then
  echo "Using Python: $PY"
else
  if [[ "$(uname -s)" == Darwin ]]; then
    echo "Python 3.10+ not found. Installing..."
    install_mac_python_ffmpeg
    PY="$(find_python)" || {
      echo "Python 3.10+ still not found after Homebrew install."
      exit 1
    }
    echo "Using Python: $PY"
  else
    echo "Python 3.10+ is required. Install it, then re-run this script."
    echo "  https://www.python.org/downloads/"
    exit 1
  fi
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg is needed for transcripts (Whisper)."
  if [[ "$(uname -s)" == Darwin ]]; then
    ensure_homebrew || exit 1
    echo "Installing ffmpeg with Homebrew..."
    brew install ffmpeg
    hash -r
  elif command -v brew >/dev/null 2>&1; then
    echo "Installing ffmpeg with Homebrew..."
    brew install ffmpeg
    hash -r
  else
    echo "Install ffmpeg, then run this script again."
    echo "  https://ffmpeg.org/download.html"
    exit 1
  fi
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg is still not on PATH after install."
  exit 1
fi

ver="$("$PY" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
echo "Creating virtualenv in .venv (Python $ver)..."
"$PY" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install -U pip
echo "Installing ptsx + Whisper (first time can take several minutes)..."
python -m pip install -e .
echo "Downloading Whisper base model (one-time, ~150 MB)..."
python -c "import whisper; whisper.load_model('base', device='cpu')"

ptsx -h >/dev/null
if [[ "$(uname -s)" == Darwin ]]; then
  python - <<'PY'
from pathlib import Path
from ptsx.macos_bundle import write_ptsx_app
app = write_ptsx_app(Path(".").resolve())
print(f"Created {app}")
PY
fi
echo
echo "Installed. On a Mac, double-click PTSX.app in this folder."
echo "Or in Terminal:"
echo "  cd \"$(pwd)\""
echo "  source .venv/bin/activate"
echo "  ptsx app"
echo
echo "CLI:"
echo "  ptsx gate --user /path/to/USER.wav --assistant /path/to/ASSISTANT.wav --outdir ./out --transcribe"
echo "  ptsx clean-cut --user /path/to/USER.wav --assistant /path/to/ASSISTANT.wav --outdir ./out --transcribe"
echo
echo "In Cursor, trigger the ptsx-standard-cut or ptsx-clean-cut project skill."
echo "Later updates: git pull, then re-run ./install.sh only if Python dependencies changed."
