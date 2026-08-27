from fastapi.testclient import TestClient

from app.main import app


def test_health_without_crashing():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code in (200, 503)


def test_analyze_rejects_empty():
    client = TestClient(app)
    response = client.post(
        "/api/analyze",
        files={"file": ("x.txt", b"not-an-image", "text/plain")},
    )
    assert response.status_code in (400, 503)


def test_missing_analysis():
    client = TestClient(app)
    response = client.get("/api/analyses/does-not-exist")
    assert response.status_code == 404
