from backend.app.core.config import settings
from backend.app.services.auth_service import get_auth_store, reset_auth_store


def test_request_code_and_login(client):
    reset_auth_store()
    object.__setattr__(settings, "ENVIRONMENT", "development")
    object.__setattr__(settings, "TEST_PHONE_NUMBERS", {"13900000001": "123456"})

    code_response = client.post("/api/v1/auth/code", json={"phone": "13900000001"})
    assert code_response.status_code == 200
    assert code_response.json()["sent"] is False

    login_response = client.post("/api/v1/auth/login", json={"phone": "13900000001", "code": "123456"})
    assert login_response.status_code == 200
    assert "access_token" in login_response.json()
    assert "refresh_token" in login_response.json()

    reset_auth_store()
    object.__setattr__(settings, "TEST_PHONE_NUMBERS", {})


def test_invalid_code_returns_400(client):
    reset_auth_store()
    object.__setattr__(settings, "ENVIRONMENT", "development")
    object.__setattr__(settings, "TEST_PHONE_NUMBERS", {"13900000001": "123456"})
    client.post("/api/v1/auth/code", json={"phone": "13900000001", "client_id": "browser-a"})

    response = client.post(
        "/api/v1/auth/login",
        json={"phone": "13900000001", "code": "000000", "client_id": "browser-a"},
    )

    assert response.status_code == 400
    reset_auth_store()
    object.__setattr__(settings, "TEST_PHONE_NUMBERS", {})




def test_development_auth_bypass_creates_placeholder_user(client):
    reset_auth_store()
    object.__setattr__(settings, "ENVIRONMENT", "development")
    object.__setattr__(settings, "DEV_AUTH_BYPASS", True)

    response = client.get(
        "/api/v1/users/real-rag-e2e-user/memories",
        headers={"X-Dev-User-Id": "real-rag-e2e-user"},
    )

    assert response.status_code == 200
    assert get_auth_store().get_user_by_id("real-rag-e2e-user") is not None


def test_production_fixed_code_config_returns_403(client):
    reset_auth_store()
    object.__setattr__(settings, "ENVIRONMENT", "production")
    object.__setattr__(settings, "TEST_PHONE_NUMBERS", {"13900000001": "123456"})

    response = client.post("/api/v1/auth/code", json={"phone": "13900000001"})

    assert response.status_code == 403
    reset_auth_store()
    object.__setattr__(settings, "ENVIRONMENT", "development")
    object.__setattr__(settings, "TEST_PHONE_NUMBERS", {})
