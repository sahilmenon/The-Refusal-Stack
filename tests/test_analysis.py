import pytest
from refusal_stack.attacks.analysis import compute_headroom


def test_compute_headroom_positive():
    result = compute_headroom(gcg_asr=0.8, pair_asr=0.5)
    assert abs(result["headroom"] - 0.3) < 1e-6
    assert "interpretation" in result
    assert "advantage" in result["interpretation"]


def test_compute_headroom_negative():
    result = compute_headroom(gcg_asr=0.4, pair_asr=0.6)
    assert result["headroom"] < 0
    assert "interpretation" in result
