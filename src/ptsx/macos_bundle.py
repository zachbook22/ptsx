"""Write a double-clickable PTSX.app that launches the Tk window from .venv."""

from __future__ import annotations

import shutil
import stat
import subprocess
from pathlib import Path

INFO_PLIST = """\
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key>
  <string>PTSX</string>
  <key>CFBundleDisplayName</key>
  <string>PTSX</string>
  <key>CFBundleIdentifier</key>
  <string>local.ptsx.app</string>
  <key>CFBundleInfoDictionaryVersion</key>
  <string>6.0</string>
  <key>CFBundleVersion</key>
  <string>0.1.0</string>
  <key>CFBundleShortVersionString</key>
  <string>0.1.0</string>
  <key>CFBundleExecutable</key>
  <string>PTSX</string>
  <key>CFBundlePackageType</key>
  <string>APPL</string>
  <key>LSMinimumSystemVersion</key>
  <string>12.0</string>
  <key>NSHighResolutionCapable</key>
  <true/>
</dict>
</plist>
"""

MACHO_MAGICS = (
    b"\xca\xfe\xba\xbe",  # fat
    b"\xcf\xfa\xed\xfe",  # 64-bit LE
    b"\xfe\xed\xfa\xcf",  # 64-bit BE
    b"\xce\xfa\xed\xfe",  # 32-bit LE
)


def _here() -> Path:
    return Path(__file__).resolve().parent


def _stub_bin() -> Path:
    return _here() / "macos_stub"


def _stub_src() -> Path:
    return _here() / "macos_launcher.c"


def is_macho(path: Path) -> bool:
    try:
        magic = path.read_bytes()[:4]
    except OSError:
        return False
    return magic in MACHO_MAGICS


def compile_macos_stub(dest: Path) -> None:
    """Compile a universal Mach-O launcher into dest."""
    src = _stub_src()
    if not src.is_file():
        raise FileNotFoundError(src)
    cc = shutil.which("cc") or shutil.which("clang")
    if not cc:
        raise RuntimeError("A C compiler (cc) is required to build PTSX.app")
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        cc,
        "-arch",
        "arm64",
        "-arch",
        "x86_64",
        "-mmacosx-version-min=12.0",
        "-framework",
        "CoreFoundation",
        "-o",
        str(dest),
        str(src),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        # Apple Silicon-only fallback if the x86_64 slice is unavailable.
        cmd = [
            cc,
            "-mmacosx-version-min=12.0",
            "-framework",
            "CoreFoundation",
            "-o",
            str(dest),
            str(src),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "cc failed to build PTSX launcher")
    dest.chmod(dest.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _install_stub(exe: Path) -> None:
    packaged = _stub_bin()
    if packaged.is_file() and is_macho(packaged):
        shutil.copy2(packaged, exe)
        exe.chmod(exe.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        return
    compile_macos_stub(exe)


def write_ptsx_app(project_root: Path) -> Path:
    """Create PTSX.app next to .venv. Returns the .app path."""
    root = Path(project_root).resolve()
    app = root / "PTSX.app"
    macos = app / "Contents" / "MacOS"
    macos.mkdir(parents=True, exist_ok=True)
    (app / "Contents" / "Info.plist").write_text(INFO_PLIST)
    (app / "Contents" / "PkgInfo").write_text("APPL????")
    exe = macos / "PTSX"
    _install_stub(exe)
    if not is_macho(exe):
        raise RuntimeError("PTSX.app launcher is not a Mach-O binary; macOS will not open it")
    return app
