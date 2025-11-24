import pytest
from ml_engine import MLEngine


def test_analyze_security_valid_code():
    """Test security analysis with valid code."""
    ml_engine = MLEngine()
    code = "def add(a, b):\n    return a + b"

    security_score, potential_issues, hints = ml_engine.analyze_security(code)

    assert isinstance(security_score, float)
    assert security_score >= 0
    assert isinstance(potential_issues, str)
    assert isinstance(hints, str)
    assert len(potential_issues) > 0
    assert len(hints) > 0


def test_analyze_security_insecure_code():
    """Test security analysis with potentially insecure code."""
    ml_engine = MLEngine()
    code = "def query(user_input):\n    sql = 'SELECT * FROM users WHERE name = ' + user_input\n    execute(sql)"

    security_score, potential_issues, hints = ml_engine.analyze_security(code)

    assert isinstance(security_score, float)
    assert security_score >= 0
    assert isinstance(potential_issues, str)
    assert isinstance(hints, str)
    # Insecure code should have issues detected
    assert len(potential_issues) > 0
    assert len(hints) > 0
