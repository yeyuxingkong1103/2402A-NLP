from fastapi.testclient import TestClient

from backend.app.main import app


def test_frontend_root_serves_page():
    client = TestClient(app)

    response = client.get("/")

    assert response.status_code == 200
    assert "霓问 NEON-ASK" in response.text
