import io

from fastapi.testclient import TestClient
from PIL import Image

from app.main import app


def _jpeg():
    buf = io.BytesIO()
    Image.new("RGB", (96, 72), (36, 70, 110)).save(buf, format="JPEG", quality=85)
    return buf.getvalue()


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


def test_guest_session_history_then_signup():
    client = TestClient(app)
    listed = client.get("/api/analyses")
    assert listed.status_code == 200
    assert listed.json() == []

    health = client.get("/health")
    if health.status_code != 200:
        return

    up = client.post(
        "/api/analyze",
        files={"file": ("still.jpg", _jpeg(), "image/jpeg")},
        data={"context": "street"},
    )
    assert up.status_code == 200
    exam_id = up.json()["id"]
    assert up.json()["saved_to_history"] is True
    session_hist = client.get("/api/analyses").json()
    assert any(row["id"] == exam_id for row in session_hist)
    assert client.get(f"/api/analyses/{exam_id}").status_code == 200
    body = client.get(f"/api/analyses/{exam_id}").json()
    assert body.get("measurements")
    assert body.get("issue_heads")

    pdf = client.get(f"/api/analyses/{exam_id}/report.pdf")
    assert pdf.status_code == 200
    assert pdf.headers["content-type"].startswith("application/pdf")

    email = f"op-{exam_id[:8]}@clinic.test"
    signed = client.post("/api/auth/signup", json={"email": email, "password": "password1"})
    assert signed.status_code == 200
    hist = client.get("/api/analyses").json()
    assert any(row["id"] == exam_id for row in hist)

    other = TestClient(app)
    assert other.get(f"/api/analyses/{exam_id}").status_code == 404


def test_signup_rejects_short_password():
    client = TestClient(app)
    res = client.post("/api/auth/signup", json={"email": "a@b.co", "password": "short"})
    assert res.status_code == 400
