from datetime import timedelta

import pytest
from jose import jwt

from backend.app.core.config import settings
from backend.app.core.security import hash_refresh_token, utc_now
from backend.app.services import auth_service
from backend.app.services.auth_service import (
    AuthServiceError,
    get_auth_store,
    login_with_code,
    refresh_access_token,
    request_login_code,
    reset_auth_store,
    revoke_refresh_token,
)


@pytest.fixture
def auth_settings(monkeypatch):
    monkeypatch.setenv("APP_MASTER_KEY", "test-master-key-32-bytes-minimum")
    monkeypatch.setenv("APP_HMAC_KEY", "test-hmac-key-32-bytes-minimum")
    object.__setattr__(settings, "ENVIRONMENT", "development")
    object.__setattr__(settings, "TEST_PHONE_NUMBERS", {})
    reset_auth_store()
    yield settings
    reset_auth_store()
    object.__setattr__(settings, "ENVIRONMENT", "development")
    object.__setattr__(settings, "TEST_PHONE_NUMBERS", {})


def _login_fixed_phone(client_id: str) -> str:
    request_login_code("13900000001", client_id=client_id)
    return login_with_code("13900000001", "123456", client_id=client_id, user_agent="pytest").refresh_token


def test_preconfigured_test_phone_uses_fixed_code(auth_settings):
    auth_settings.TEST_PHONE_NUMBERS = {"13900000001": "123456"}

    result = request_login_code("13900000001", client_id="browser-a")

    assert result.sent is False
    assert result.expires_in_seconds == 300


def test_login_requires_correct_code(auth_settings):
    auth_settings.TEST_PHONE_NUMBERS = {"13900000001": "123456"}
    request_login_code("13900000001", client_id="browser-a")

    tokens = login_with_code("13900000001", "123456", client_id="browser-a", user_agent="pytest")

    assert tokens.access_token
    assert tokens.refresh_token
    assert tokens.token_type == "bearer"


def test_wrong_code_locks_for_15_minutes_and_then_allows_new_code(auth_settings, monkeypatch):
    auth_settings.TEST_PHONE_NUMBERS = {"13900000001": "123456"}
    current_time = utc_now()
    monkeypatch.setattr(auth_service, "utc_now", lambda: current_time)
    request_login_code("13900000001", client_id="browser-a")

    for _ in range(5):
        with pytest.raises(AuthServiceError):
            login_with_code("13900000001", "000000", client_id="browser-a", user_agent="pytest")

    with pytest.raises(AuthServiceError, match="locked"):
        login_with_code("13900000001", "123456", client_id="browser-a", user_agent="pytest")
    with pytest.raises(AuthServiceError, match="locked"):
        request_login_code("13900000001", client_id="browser-a")

    current_time = current_time + timedelta(seconds=899)
    with pytest.raises(AuthServiceError, match="locked"):
        request_login_code("13900000001", client_id="browser-a")

    current_time = current_time + timedelta(seconds=1)
    result = request_login_code("13900000001", client_id="browser-a")
    tokens = login_with_code("13900000001", "123456", client_id="browser-a", user_agent="pytest")

    assert result.sent is False
    assert tokens.access_token


def test_refresh_and_revoke_token(auth_settings):
    auth_settings.TEST_PHONE_NUMBERS = {"13900000001": "123456"}
    request_login_code("13900000001", client_id="browser-a")
    tokens = login_with_code("13900000001", "123456", client_id="browser-a", user_agent="pytest")

    refreshed = refresh_access_token(tokens.refresh_token)
    assert refreshed.access_token

    revoke_refresh_token(refreshed.refresh_token)
    with pytest.raises(AuthServiceError):
        refresh_access_token(refreshed.refresh_token)


def test_token_ttl_and_max_five_devices_revoke_oldest(auth_settings):
    auth_settings.TEST_PHONE_NUMBERS = {"13900000001": "123456"}

    refresh_tokens = [_login_fixed_phone(f"browser-{index}") for index in range(6)]
    store = get_auth_store()
    sessions = [store.get_session_by_hash(hash_refresh_token(token)) for token in refresh_tokens]
    active_sessions = [session for session in sessions if session and not session.revoked]

    assert sessions[0] is not None and sessions[0].revoked is True
    assert len(active_sessions) == 5
    assert sessions[-1] is not None
    assert sessions[-1].expires_at - sessions[-1].login_at == timedelta(days=30)


def test_access_token_expires_in_two_hours(auth_settings):
    auth_settings.TEST_PHONE_NUMBERS = {"13900000001": "123456"}
    request_login_code("13900000001", client_id="browser-a")

    tokens = login_with_code("13900000001", "123456", client_id="browser-a", user_agent="pytest")
    payload = jwt.get_unverified_claims(tokens.access_token)

    assert payload["exp"] - int(utc_now().timestamp()) <= 7200
    assert payload["exp"] - int(utc_now().timestamp()) > 7190


def test_production_rejects_fixed_test_phone_config(auth_settings):
    auth_settings.ENVIRONMENT = "production"
    auth_settings.TEST_PHONE_NUMBERS = {"13900000001": "123456"}

    with pytest.raises(AuthServiceError, match="disabled in production"):
        request_login_code("13900000001", client_id="browser-a")
