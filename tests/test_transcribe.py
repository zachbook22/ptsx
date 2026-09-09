"""Whisper is a core install dependency."""

from ptsx.transcribe import whisper_available


def test_whisper_available_is_bool():
    assert isinstance(whisper_available(), bool)
