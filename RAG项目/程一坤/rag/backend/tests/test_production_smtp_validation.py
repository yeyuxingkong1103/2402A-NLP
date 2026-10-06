"""生产环境 SMTP 启动期校验测试（批次 3-3）。

要求：ENVIRONMENT=production 且缺 SMTP_* → Settings.from_environment()
在应用启动期即报错，报错只写缺失的键名，不打印任何值。
"""
import pytest

from app.core.config import Settings


@pytest.fixture()
def clean_production_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """构造一个 production 环境：业务必填项齐全，SMTP_* 全部清空。

    屏蔽真实 .env 的注入（load_environment_file 替身为空操作），
    否则本机 .env 的 SMTP_* 会被重新读入，无法模拟"缺失"。
    """
    monkeypatch.setattr("app.core.config.load_environment_file", lambda *a, **k: 0)
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("REDIS_URL", "redis://prod-redis:6379/0")
    monkeypatch.setenv("EMBEDDING_API_KEY", "fake-key")
    monkeypatch.setenv("RERANKER_API_KEY", "fake-key")
    monkeypatch.setenv("LLM_API_BASE_URL", "https://llm.example.com/v1")
    monkeypatch.setenv("LLM_API_KEY", "fake-key")
    monkeypatch.setenv("LLM_MODEL", "fake-model")
    for key in ("SMTP_HOST", "SMTP_USERNAME", "SMTP_PASSWORD", "SMTP_FROM_EMAIL"):
        monkeypatch.delenv(key, raising=False)


def test_production_without_smtp_fails_at_startup(clean_production_env) -> None:
    """production 缺 SMTP_* → from_environment() 启动期抛错，报错只含键名。"""
    with pytest.raises(ValueError) as exc_info:
        Settings.from_environment()
    message = str(exc_info.value)
    assert "SMTP_HOST" in message
    assert "SMTP_USERNAME" in message
    # 报错只写键名，不打印任何配置值（脱敏）
    assert "wjq" not in message
    assert "@" not in message  # 邮箱地址/主机名不应出现在报错里


def test_production_with_smtp_configured_passes(clean_production_env, monkeypatch) -> None:
    """production 且 SMTP_* 齐全 → 启动校验通过。"""
    monkeypatch.setenv("SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("SMTP_PORT", "465")
    monkeypatch.setenv("SMTP_USERNAME", "bot@example.com")
    monkeypatch.setenv("SMTP_PASSWORD", "fake-password")
    monkeypatch.setenv("SMTP_FROM_EMAIL", "bot@example.com")
    monkeypatch.setenv("SMTP_USE_SSL", "true")
    config = Settings.from_environment()
    assert config.environment == "production"
    assert config.smtp_host == "smtp.example.com"


def test_development_without_smtp_still_loads_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """development 配置读取本身仍可完成，SMTP 完整性由 mailer 创建阶段校验。"""
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.delenv("SMTP_HOST", raising=False)
    config = Settings.from_environment()
    assert config.environment == "development"
