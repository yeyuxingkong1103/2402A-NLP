from backend.app.config import get_settings


def test_runtime_uses_persistent_local_service_endpoints(monkeypatch):
    monkeypatch.setenv("MILVUS_URI", "http://127.0.0.1:19530")
    monkeypatch.setenv("MILVUS_DATABASE", "default")
    monkeypatch.setenv("EMBEDDING_DIM", "1024")
    monkeypatch.setenv("REDIS_URL", "redis://127.0.0.1:6379/0")
    monkeypatch.setenv(
        "MYSQL_URL",
        "mysql+pymysql://law_rag:law_rag_password@127.0.0.1:3306/law_rag",
    )
    get_settings.cache_clear()

    settings = get_settings()

    assert settings.milvus_uri == "http://127.0.0.1:19530"
    assert settings.milvus_database == "default"
    assert settings.embedding_dim == 1024
    assert settings.redis_url == "redis://127.0.0.1:6379/0"
    assert settings.mysql_url.endswith("@127.0.0.1:3306/law_rag")


def test_environment_changes_are_read_when_settings_are_created(monkeypatch):
    monkeypatch.setenv("MILVUS_URI", "http://milvus.example:19530")
    monkeypatch.setenv("EMBEDDING_DIM", "768")
    get_settings.cache_clear()

    settings = get_settings()

    assert settings.milvus_uri == "http://milvus.example:19530"
    assert settings.embedding_dim == 768


def test_compose_mysql_credentials_override_obsolete_legacy_url(monkeypatch):
    monkeypatch.setenv(
        "MYSQL_URL",
        "mysql+pymysql://lawrag_app:obsolete@127.0.0.1:3306/law_rag",
    )
    monkeypatch.setenv("MYSQL_APP_PASSWORD", "new password")
    get_settings.cache_clear()

    settings = get_settings()

    assert settings.mysql_url == (
        "mysql+pymysql://law_rag:new+password@127.0.0.1:3306/law_rag"
    )


def test_runtime_reads_mineru_api_settings(monkeypatch):
    monkeypatch.setenv("MINERU_ENABLED", "true")
    monkeypatch.setenv("MINERU_API_TOKEN", "test-token")
    monkeypatch.setenv("MINERU_API_BASE_URL", "https://mineru.example/api/v4")
    monkeypatch.setenv("MINERU_MODEL_VERSION", "vlm")
    monkeypatch.setenv("MINERU_TIMEOUT_SECONDS", "600")
    monkeypatch.setenv("MINERU_POLL_INTERVAL_SECONDS", "3")
    get_settings.cache_clear()

    settings = get_settings()

    assert settings.mineru_enabled is True
    assert settings.mineru_api_token == "test-token"
    assert settings.mineru_api_base_url == "https://mineru.example/api/v4"
    assert settings.mineru_model_version == "vlm"
    assert settings.mineru_timeout == 600
    assert settings.mineru_poll_interval == 3
