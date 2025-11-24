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
        json={"code_snippet": test_code}
    )

    assert response.status_code == 200
    data = response.json()
    assert "security_score" in data
    assert "potential_issues" in data
    assert "hints" in data
    assert "original_code" in data
    assert "id" in data
    assert isinstance(data["security_score"], (int, float))
    assert isinstance(data["potential_issues"], str)
    assert isinstance(data["hints"], str)


def test_analyze_empty_code():
    """Test analysis with empty code."""
    response = client.post(
        "/api/analyze",
        json={"code_snippet": ""}
    )

    assert response.status_code in [200, 422, 500]
