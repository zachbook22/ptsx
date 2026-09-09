"""Optional checks against Pro Tools gold bounces when they are on disk."""

from pathlib import Path

import numpy as np
import pytest

from ptsx.audio import load_mono

PRED = Path(__file__).resolve().parents[1] / "out"


@pytest.mark.skipif(not (PRED / "USER.wav").exists(), reason="Predicted bounces not present")
def test_pred_no_overlap():
    pu, _sr = load_mono(PRED / "USER.wav")
    pa, _ = load_mono(PRED / "ASSISTANT.wav")
    n = min(len(pu), len(pa))
    overlap = int(np.sum((np.abs(pu[:n]) > 0.0) & (np.abs(pa[:n]) > 0.0)))
    assert overlap == 0
