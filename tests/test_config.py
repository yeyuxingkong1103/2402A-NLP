"""配置加载测试。"""

from __future__ import annotations

import pytest

from role_rag.config import Config, load_config
from role_rag.errors import ConfigError


def test_defaults_and_yaml(project_root):
    config = load_config(project_root, env_file=None)
    assert config.get("app.name") == "role-rag"
    assert config.get("milvus.kb_collection") == "role_kb_chunks"
    assert config.get("retrieval.weights.dense") == pytest.approx(0.55)
    assert config.get("models.embedder.dense_dim") == 1024


def test_dot_path_fallback():
    config = Config({"a": {"b": {"c": 1}}})
    assert config.get("a.b.c") == 1
    assert config.get("a.b.missing", "default") == "default"
    assert config.get("x.y.z", None) is None


def test_env_override(project_root, monkeypatch):
    monkeypatch.setenv("ROLE_RAG_FINAL_K", "9")
    monkeypatch.setenv("ROLE_RAG_CACHE_ENABLED", "false")
    monkeypatch.setenv("ROLE_RAG_LLM_MODEL", "D:/modelscope/Qwen3-0.6B")
    config = load_config(project_root, env_file=None)
    assert config.get("retrieval.final_k") == 9
    assert config.get("retrieval.cache_enabled") is False
    assert config.llm_path().name == "Qwen3-0.6B"


def test_model_paths_from_models_root(project_root):
    config = load_config(project_root, env_file=None)
    assert config.embedder_path().name == "bge-m3"
    assert config.models_root.name == "modelscope"
    assert config.kb_dir.name == "kb"
    assert config.kb_dir.is_dir()


def test_missing_secret_raises(project_root, monkeypatch):
    config = load_config(project_root, env_file=None)
    config._data["security"]["secret"] = ""
    with pytest.raises(ConfigError):
        if not config.get("security.secret"):
            raise ConfigError("security.secret 未配置")
