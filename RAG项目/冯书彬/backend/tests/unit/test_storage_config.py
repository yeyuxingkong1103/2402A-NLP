from backend.app.core.config import AppSettings
from backend.app.core.storage import AuthStoreConfigurationError, configure_auth_store
from backend.app.services.auth_service import get_auth_store
from backend.app.services.auth_store import InMemoryAuthStore


def test_configure_auth_store_defaults_to_memory():
    configure_auth_store(AppSettings(AUTH_STORE_BACKEND="memory"))

    assert isinstance(get_auth_store(), InMemoryAuthStore)


def test_configure_sql_auth_store_requires_database_url():
    settings = AppSettings(AUTH_STORE_BACKEND="sql", DATABASE_URL="")

    try:
        configure_auth_store(settings)
    except AuthStoreConfigurationError as exc:
        assert "DATABASE_URL" in str(exc)
    else:
        raise AssertionError("sql auth store should require DATABASE_URL")


def test_configure_redis_otp_store_requires_redis_url():
    settings = AppSettings(OTP_STORE_BACKEND="redis", REDIS_URL="")

    try:
        configure_auth_store(settings)
    except AuthStoreConfigurationError as exc:
        assert "REDIS_URL" in str(exc)
    else:
        raise AssertionError("redis OTP store should require REDIS_URL")
