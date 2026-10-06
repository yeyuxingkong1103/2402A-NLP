from datetime import timedelta

from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from backend.app.core.config import settings
from backend.app.core.security import hash_refresh_token
from backend.app.repositories.auth_repository import SQLAlchemyAuthStore, create_auth_tables
from backend.app.services.auth_service import get_auth_store, login_with_code, refresh_access_token, request_login_code, set_auth_store


def _create_sql_store() -> SQLAlchemyAuthStore:
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool, future=True)
    create_auth_tables(engine)
    return SQLAlchemyAuthStore(engine)


def test_sql_auth_store_persists_user_and_session(monkeypatch):
    monkeypatch.setenv("APP_MASTER_KEY", "test-master-key-32-bytes-minimum")
    monkeypatch.setenv("APP_HMAC_KEY", "test-hmac-key-32-bytes-minimum")
    object.__setattr__(settings, "ENVIRONMENT", "development")
    object.__setattr__(settings, "TEST_PHONE_NUMBERS", {"13900000001": "123456"})
    store = _create_sql_store()
    set_auth_store(store)

    request_login_code("13900000001", client_id="browser-a")
    tokens = login_with_code("13900000001", "123456", client_id="browser-a", user_agent="pytest")
    session = store.get_session_by_hash(hash_refresh_token(tokens.refresh_token))

    assert session is not None
    assert store.get_user_by_id(tokens.user_id) is not None
    assert timedelta(days=30) - (session.expires_at - session.login_at) < timedelta(seconds=1)


def test_sql_auth_store_rotates_refresh_token(monkeypatch):
    monkeypatch.setenv("APP_MASTER_KEY", "test-master-key-32-bytes-minimum")
    monkeypatch.setenv("APP_HMAC_KEY", "test-hmac-key-32-bytes-minimum")
    object.__setattr__(settings, "ENVIRONMENT", "development")
    object.__setattr__(settings, "TEST_PHONE_NUMBERS", {"13900000001": "123456"})
    store = _create_sql_store()
    set_auth_store(store)

    request_login_code("13900000001", client_id="browser-a")
    tokens = login_with_code("13900000001", "123456", client_id="browser-a", user_agent="pytest")
    old_hash = hash_refresh_token(tokens.refresh_token)
    refreshed = refresh_access_token(tokens.refresh_token)

    assert store.get_session_by_hash(old_hash) is None
    assert store.get_session_by_hash(hash_refresh_token(refreshed.refresh_token)) is not None


def test_sql_auth_store_deletes_user_and_sessions(monkeypatch):
    monkeypatch.setenv("APP_MASTER_KEY", "test-master-key-32-bytes-minimum")
    monkeypatch.setenv("APP_HMAC_KEY", "test-hmac-key-32-bytes-minimum")
    object.__setattr__(settings, "ENVIRONMENT", "development")
    object.__setattr__(settings, "TEST_PHONE_NUMBERS", {"13900000001": "123456"})
    store = _create_sql_store()
    set_auth_store(store)

    request_login_code("13900000001", client_id="browser-a")
    tokens = login_with_code("13900000001", "123456", client_id="browser-a", user_agent="pytest")
    token_hash = hash_refresh_token(tokens.refresh_token)
    get_auth_store().delete_user(tokens.user_id)

    assert store.get_user_by_id(tokens.user_id) is None
    assert store.is_user_deleted(tokens.user_id) is True
    assert store.get_session_by_hash(token_hash) is None
