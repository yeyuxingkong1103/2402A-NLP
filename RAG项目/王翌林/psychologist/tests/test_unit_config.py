"""单元测试：config 配置读取、属性解析与启动安全校验（全部不触网）。"""
import pytest

from src.core.config import Settings, settings


def test_env_values_loaded():
    """关键配置必须从 .env 正确读取（阶段 2 修复点）。"""
    assert settings.db_port == 3307
    assert settings.db_name == "rag_roleplay"
    assert settings.db_user == "dev"
    assert settings.redis_port == 6379
    assert settings.milvus_port == 19530
    assert settings.milvus_collection == "persona_knowledge"
    assert settings.milvus_memory_collection == "user_long_term_memory"
    assert settings.embedding_dim == 1024
    assert "bge-m3" in settings.embedding_model_path
    assert "bge-reranker-v2-m3" in settings.reranker_model_path
    assert settings.llm_model == "deepseek-flash"
    assert settings.llm_thinking == "disabled"


def test_env_secrets_not_placeholder():
    """真实 .env 的密钥不应是占位符（示例文件才允许占位符）。"""
    assert settings.jwt_secret_key not in ("", "change_me")
    assert settings.admin_password != ""


def test_cors_origins_parse():
    assert isinstance(settings.cors_origins, list)
    assert all(o.startswith("http") for o in settings.cors_origins)


def test_emergency_phones_parse():
    assert settings.crisis_hotline == "12356"
    assert "120" in settings.emergency_phones and "110" in settings.emergency_phones


def test_validate_security_ok():
    settings.validate_security()  # 不抛异常即通过


def _isolated_settings(**kwargs) -> Settings:
    """构造脱离 .env 的独立实例，测校验分支。"""
    base = dict(jwt_secret_key="x" * 40, admin_password="admin123456")
    base.update(kwargs)
    return Settings(_env_file=None, **base)


def test_validate_rejects_placeholder_jwt():
    with pytest.raises(RuntimeError, match="JWT_SECRET_KEY"):
        _isolated_settings(jwt_secret_key="change_me").validate_security()


def test_validate_rejects_weak_jwt():
    with pytest.raises(RuntimeError, match="强度不足"):
        _isolated_settings(jwt_secret_key="short").validate_security()


def test_validate_rejects_missing_admin_password():
    with pytest.raises(RuntimeError, match="ADMIN_PASSWORD"):
        _isolated_settings(admin_password="").validate_security()
