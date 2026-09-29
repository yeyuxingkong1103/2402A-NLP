from backend.app.core.config import AppSettings, validate_production_settings
import pytest


def test_production_environment_disables_development_auth_bypass_by_default():
    settings = AppSettings(ENVIRONMENT="production")

    assert settings.DEV_AUTH_BYPASS is False


def test_production_settings_require_persistent_backends_and_secrets():
    with pytest.raises(RuntimeError, match="AUTH_STORE_BACKEND must be sql"):
        validate_production_settings(AppSettings(ENVIRONMENT="production"))


def test_production_settings_accept_complete_secure_configuration():
    settings = AppSettings(
        ENVIRONMENT="production",
        AUTH_STORE_BACKEND="sql",
        OTP_STORE_BACKEND="redis",
        REQUEST_CONTROL_BACKEND="redis",
        DATABASE_URL="mysql+pymysql://app:password@mysql/legal_rag",
        REDIS_URL="redis://redis:6379/0",
        MILVUS_URI="http://milvus:19530",
        CELERY_BROKER_URL="redis://redis:6379/0",
        CELERY_RESULT_BACKEND="redis://redis:6379/1",
        APP_MASTER_KEY="m" * 32,
        APP_HMAC_KEY="h" * 32,
        DEEPSEEK_API_KEY="deepseek-key",
        SERVE_FRONTEND=False,
        CORS_ORIGINS="https://legal.example.com",
    )

    validate_production_settings(settings)


def test_non_production_settings_skip_production_validation():
    validate_production_settings(AppSettings(ENVIRONMENT="development"))


def test_model_paths_can_be_injected_for_deployment():
    settings = AppSettings(
        BGE_M3_MODEL_PATH="/opt/models/bge-m3",
        BGE_RERANKER_MODEL_PATH="/opt/models/bge-reranker-large",
    )

    assert settings.model_paths == {
        "bge_m3": "/opt/models/bge-m3",
        "bge_reranker": "/opt/models/bge-reranker-large",
    }


def test_langchain_rag_framework_is_the_default():
    assert AppSettings().RAG_FRAMEWORK == "langchain"


def test_legacy_rag_framework_remains_available_as_fallback():
    assert AppSettings(RAG_FRAMEWORK="legacy").RAG_FRAMEWORK == "legacy"


def test_langchain_rag_framework_can_be_enabled_explicitly():
    assert AppSettings(RAG_FRAMEWORK="langchain").RAG_FRAMEWORK == "langchain"
