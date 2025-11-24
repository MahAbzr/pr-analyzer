import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_health_check():
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_analyze_code():
    """Test code analysis endpoint."""
    test_code = "def test():\n    x = 1 / 0\n    return x"

    response = client.post(
        "/api/analyze",
        json={"code": test_code}
    )

    assert response.status_code == 200
    data = response.json()
    assert "risk_score_before" in data
    assert "risk_score_after" in data
    assert "fixed_code" in data


def test_analyze_empty_code():
    """Test analysis with empty code."""
    response = client.post(
        "/api/analyze",
        json={"code": ""}
    )

    assert response.status_code in [200, 400]
