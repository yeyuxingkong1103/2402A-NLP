from backend.app.core.startup_checks import DependencyStatus


def test_live_health_returns_alive(client):
    response = client.get("/health/live")

    assert response.status_code == 200
    assert response.json()["status"] == "alive"


def test_ready_health_has_dependencies(client, monkeypatch):
    def fake_startup_checks(_settings):
        return [DependencyStatus("mysql", True), DependencyStatus("milvus", False, "missing uri")]

    monkeypatch.setattr("backend.app.api.v1.health.run_startup_checks", fake_startup_checks)

    response = client.get("/health/ready")

    assert response.status_code == 503
    body = response.json()
    assert body["ready"] is False
    assert body["dependencies"] == [
        {"name": "mysql", "ok": True, "detail": None},
        {"name": "milvus", "ok": False, "detail": "missing uri"},
    ]


def test_frontend_login_page_is_served_by_backend(client):
    response = client.get("/frontend/pages/login.html")

    assert response.status_code == 200
    assert "法律客服助手" in response.text


def test_local_frontend_origin_is_allowed(client):
    response = client.options(
        "/api/v1/auth/code",
        headers={
            "Origin": "http://127.0.0.1:3010",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://127.0.0.1:3010"
