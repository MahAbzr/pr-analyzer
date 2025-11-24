import pytest
from ml_engine import extract_features, calculate_score_and_get_fixed_code

def test_extract_features_basic():
    """Test feature extraction with simple code."""
    code = "def foo():\n    x = 1\n    return x"
    features = extract_features(code)

    assert features is not None
    assert len(features) == 768  # CodeBERT embedding size


def test_extract_features_empty():
    """Test feature extraction with empty code."""
    features = extract_features("")
    assert features is not None


def test_calculate_score_valid_code():
    """Test score calculation with valid code."""
    code = "def add(a, b):\n    return a + b"
    pred_before, pred_after, fixed = calculate_score_and_get_fixed_code(code)

    assert isinstance(pred_before, float)
    assert isinstance(pred_after, float)
    assert isinstance(fixed, str)


def test_calculate_score_buggy_code():
    """Test score calculation with potentially buggy code."""
    code = "def div(a, b):\n    return a / b"  # No zero check
    pred_before, pred_after, fixed = calculate_score_and_get_fixed_code(code)

    assert pred_before >= 0
    assert pred_after >= 0
