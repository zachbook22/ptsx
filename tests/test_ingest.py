from __future__ import annotations

from pathlib import Path

from ptsx.ingest import (
    choose_mode,
    find_pair,
    format_dispatcher_message,
    write_editor_txt,
    write_ingest_summary,
)


def test_find_pair(tmp_path: Path) -> None:
    (tmp_path / "TASK12_USER.wav").write_bytes(b"RIFF")
    (tmp_path / "TASK12_ASSISTANT.wav").write_bytes(b"RIFF")
    user, assistant = find_pair(tmp_path)
    assert user.name == "TASK12_USER.wav"
    assert assistant.name == "TASK12_ASSISTANT.wav"


def test_choose_mode() -> None:
    assert choose_mode(already_enhanced=True, plugins_present=True) == "gate"
    assert choose_mode(already_enhanced=False, plugins_present=True) == "clean-cut"
    assert choose_mode(already_enhanced=False, plugins_present=False) == "gate"


def test_editor_txt_and_dispatcher(tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    (out / "script.csv").write_text(
        "start_s,end_s,speaker,text\n"
        "0.0,2.0,USER,hello\n"
        "2.5,4.0,ASSISTANT,hi\n",
        encoding="utf-8",
    )
    (out / "cuts.csv").write_text(
        "track,reason,start_s,end_s,duration_s\n"
        "USER,overlap_bleed,1.0,1.2,0.2\n"
        "ASSISTANT,overlap_bleed,3.0,3.4,0.4\n"
        "USER,hangover,4.0,4.5,0.5\n",
        encoding="utf-8",
    )
    (out / "clips.csv").write_text("clip_id,track\n1,USER\n2,ASSISTANT\n", encoding="utf-8")
    (out / "USER.wav").write_bytes(b"x")
    (out / "ASSISTANT.wav").write_bytes(b"x")
    (out / "USER_REMOVED.wav").write_bytes(b"x")
    (out / "ASSISTANT_REMOVED.wav").write_bytes(b"x")

    editor = write_editor_txt(
        out,
        task_id="TASK12",
        mode="clean-cut",
        already_enhanced=False,
        plugins_present=True,
        transcribe=True,
    )
    text = editor.read_text(encoding="utf-8")
    assert "TASK12" in text
    assert "clean-cut" in text
    assert "USER_REMOVED.wav" in text
    assert "Hangover = mute" in text
    assert "Do not run Adobe Enhance and then clean-cut" in text

    summary = write_ingest_summary(
        out,
        task_id="TASK12",
        mode="clean-cut",
        already_enhanced=False,
        plugins_present=True,
        transcribe=True,
        user_in=tmp_path / "TASK12_USER.wav",
        assistant_in=tmp_path / "TASK12_ASSISTANT.wav",
        extra={"whisper_model": "base"},
    )
    data = __import__("json").loads(summary.read_text(encoding="utf-8"))
    assert data["task_id"] == "TASK12"
    assert data["mode"] == "clean-cut"
    assert data["dispatcher"]["clip_count"] == 2
    assert data["dispatcher"]["cut_count"] == 3
    assert data["dispatcher"]["overlap_bleed_count"] == 2
    msg = format_dispatcher_message(data)
    assert "TASK12" in msg
    assert "clean-cut" in msg
    assert "2 clips" in msg
    assert "do not attach hour-long WAVs" in msg
    assert data["grokbot"]["attach_wavs"] is False
    assert data["grokbot"]["render_audio"] is False


def test_find_pair_requires_both_roles(tmp_path: Path) -> None:
    (tmp_path / "TASK12_USER.wav").write_bytes(b"RIFF")
    try:
        find_pair(tmp_path)
        raise AssertionError("expected SystemExit")
    except SystemExit as exc:
        assert "TASKID_USER" in str(exc)


def test_task_id_from_pair() -> None:
    from ptsx.ingest import task_id_from_pair

    user = Path("/share/TASK12_USER.wav")
    assistant = Path("/share/TASK12_ASSISTANT.wav")
    assert task_id_from_pair(user, assistant) == "TASK12"


def test_ingest_status_cli(tmp_path: Path, capsys) -> None:
    from ptsx.cli import main
    from ptsx.ingest import write_ingest_summary

    out = tmp_path / "out"
    out.mkdir()
    (out / "clips.csv").write_text("speaker\nUSER\nASSISTANT\n", encoding="utf-8")
    (out / "cuts.csv").write_text("reason\noverlap_bleed\n", encoding="utf-8")
    write_ingest_summary(
        out,
        task_id="TASK99",
        mode="gate",
        already_enhanced=True,
        plugins_present=False,
        transcribe=False,
    )
    assert main(["ingest-status", "--outdir", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "TASK99" in printed
    assert "gate" in printed
    assert "do not attach hour-long WAVs" in printed


def test_ingest_pair_routes_already_enhanced_to_gate(tmp_path: Path, monkeypatch) -> None:
    from ptsx import ingest as ingest_mod

    user = tmp_path / "TASK7_USER.wav"
    assistant = tmp_path / "TASK7_ASSISTANT.wav"
    user.write_bytes(b"RIFF")
    assistant.write_bytes(b"RIFF")
    out = tmp_path / "out"
    called: dict[str, object] = {}

    def fake_gate(user_path, assistant_path, outdir, cfg):
        called["gate"] = True
        called["conform"] = cfg.conform
        (Path(outdir) / "clips.csv").write_text("speaker\nUSER\n", encoding="utf-8")
        (Path(outdir) / "cuts.csv").write_text("reason\n", encoding="utf-8")
        return {"user": Path(outdir) / "USER.wav"}

    def fake_clean(*_a, **_k):
        raise AssertionError("clean-cut must not run on already-enhanced files")

    monkeypatch.setattr(ingest_mod, "gate_files", fake_gate)
    monkeypatch.setattr(ingest_mod, "clean_cut_files", fake_clean)
    paths = ingest_mod.ingest_pair(
        user,
        assistant,
        out,
        already_enhanced=True,
        transcribe=False,
        plugins_present=True,
    )
    assert called["gate"] is True
    assert called["conform"] is False
    assert paths["editor"].name == "EDITOR.txt"
    assert paths["summary"].name == "ingest_summary.json"


def test_ingest_pair_uses_clean_cut_when_plugins_present(tmp_path: Path, monkeypatch) -> None:
    from ptsx import ingest as ingest_mod

    user = tmp_path / "TASK8_USER.wav"
    assistant = tmp_path / "TASK8_ASSISTANT.wav"
    user.write_bytes(b"RIFF")
    assistant.write_bytes(b"RIFF")
    out = tmp_path / "out"
    called = {"clean": False}

    def fake_clean(user_path, assistant_path, outdir, cfg):
        called["clean"] = True
        (Path(outdir) / "clips.csv").write_text("speaker\nUSER\nASSISTANT\n", encoding="utf-8")
        (Path(outdir) / "cuts.csv").write_text("reason\noverlap_bleed\n", encoding="utf-8")
        return {"user": Path(outdir) / "USER.wav"}

    def fake_gate(*_a, **_k):
        raise AssertionError("gate must not run when plugins are present")

    monkeypatch.setattr(ingest_mod, "clean_cut_files", fake_clean)
    monkeypatch.setattr(ingest_mod, "gate_files", fake_gate)
    ingest_mod.ingest_pair(
        user,
        assistant,
        out,
        already_enhanced=False,
        transcribe=True,
        plugins_present=True,
    )
    assert called["clean"] is True
