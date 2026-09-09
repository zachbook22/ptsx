from pathlib import Path

from ptsx.macos_bundle import is_macho, write_ptsx_app


def test_write_ptsx_app(tmp_path: Path):
    app = write_ptsx_app(tmp_path)
    exe = app / "Contents" / "MacOS" / "PTSX"
    assert app.name == "PTSX.app"
    assert exe.is_file()
    assert is_macho(exe)
    plist = (app / "Contents" / "Info.plist").read_text()
    assert "CFBundleExecutable" in plist
    assert "<string>PTSX</string>" in plist
