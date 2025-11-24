import pytest
from ml_engine import MLEngine


def test_calculate_score_valid_code():

    ml_engine = MLEngine()
    """Test score calculation with valid code."""
    code = "def add(a, b):\n    return a + b"
    pred_before, pred_after, fixed = ml_engine.calculate_score_and_get_fixed_code(code)

    assert isinstance(pred_before, float)
    assert isinstance(pred_after, float)
    assert isinstance(fixed, str)
