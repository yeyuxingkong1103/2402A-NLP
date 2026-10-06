from fastapi.testclient import TestClient

from backend.app.config import Settings, env_bool, env_int
from backend.app.main import app


def test_settings_use_the_same_defaults_as_the_old_project() -> None:
    settings = Settings()

    assert settings.host == "0.0.0.0"
    assert settings.port == 7294
    assert settings.debug is False
    assert settings.expose_docs is True
    assert settings.log_level == "INFO"
    assert settings.embedding_retry_count == 1
    assert settings.embedding_retry_delay == 1.0
    assert settings.embedding_failure_cooldown == 30
    assert settings.embedding_connect_timeout == 5


def test_environment_conversion_helpers_are_easy_to_understand(monkeypatch) -> None:
    monkeypatch.setenv("EXAMPLE_NUMBER", "12")
    monkeypatch.setenv("EXAMPLE_SWITCH", "yes")

    assert env_int("EXAMPLE_NUMBER", 1) == 12
    assert env_bool("EXAMPLE_SWITCH", False) is True


def test_health_uses_the_old_project_response_format() -> None:
    class HealthyService:
        def __init__(self, payload):
            self.payload = payload

        def health(self):
            return self.payload

    app.state.services = {
        "mysql": HealthyService({"connected": True}),
        "redis": HealthyService({"connected": True}),
        "milvus": HealthyService({
            "connected": True,
            "public_collections": ["civil_code_articles"],
            "public_collection_counts": {"civil_code_articles": 1260},
            "indexed_records": 1260,
        }),
        "model": HealthyService({
            "llm_provider": "DeepSeek",
            "llm_model": "deepseek-chat",
            "embedding_model": "BAAI/bge-m3",
            "reranker_model": "BAAI/bge-reranker-v2-m3",
            "web": "configured",
        }),
        "workspace": HealthyService({"connected": True}),
    }

    response = TestClient(app).get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert payload["data"]["status"] == "ok"
    assert payload["data"]["milvus"]["public_collection_counts"] == {
        "civil_code_articles": 1260
    }
    assert payload["data"]["indexed_records"] == 1260


def test_health_is_degraded_after_embedding_failure() -> None:
    class HealthyService:
        def __init__(self, payload):
            self.payload = payload

        def health(self):
            return self.payload

    app.state.services = {
        "mysql": HealthyService({"connected": True}),
        "redis": HealthyService({"connected": True}),
        "milvus": HealthyService({
            "connected": True,
            "public_collections": ["civil_code_articles"],
            "indexed_records": 1260,
        }),
        "model": HealthyService({
            "llm_provider": "DeepSeek",
            "llm_model": "deepseek-chat",
            "embedding_model": "BAAI/bge-m3",
            "embedding_last_error": "network unavailable",
            "reranker_model": "BAAI/bge-reranker-v2-m3",
            "web": "offline",
        }),
        "workspace": HealthyService({"connected": True}),
    }

    payload = TestClient(app).get("/health").json()["data"]

    assert payload["status"] == "degraded"
    assert payload["rag"] == "degraded"
