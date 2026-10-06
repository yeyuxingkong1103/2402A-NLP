"""批次 38：集成测试验证码旁路的守卫测试。

背景：集成测试（scripts/e2e/）起独立进程跑真实 HTTP 链路，读不到进程内验证码、
也不便走 SMTP 收信，因此加入「ENVIRONMENT=test 时接受固定码 000000」的旁路。

本文件锁死三件事：
1. test 环境：固定码放行，且打 INFO 日志（便于排查）；
2. development / production：固定码**必须被拒**（这是本次最重要的守卫）；
3. 旁路不影响真实码：test 环境下非固定码仍走 store 校验（错码照样拒）。
"""

import logging
from dataclasses import replace

import pytest

from app.auth import service as auth_service_module
from app.auth.mailer import InMemoryMailer
from app.auth.service import (
    INTEGRATION_TEST_FIXED_CODE,
    AuthService,
    InMemoryAuthStore,
)
from app.auth.session_store import SessionStore
from app.core.config import settings
from tests.conftest import FakeRedis


def _build_service() -> AuthService:
    return AuthService(
        store=InMemoryAuthStore(),
        mailer=InMemoryMailer(),
        session_store=SessionStore(FakeRedis(), 3600),
    )


@pytest.fixture
def environment(monkeypatch):
    """按用例设置 settings.environment。

    Settings 是 frozen dataclass，不能改属性——沿用 test_synonym_expansion 的范式：
    替换被测模块内的 settings 引用（dataclasses.replace 造新实例）。
    """

    def _set(value: str) -> None:
        monkeypatch.setattr(
            auth_service_module, "settings", replace(settings, environment=value)
        )

    return _set


def test_test_env_accepts_fixed_code(environment):
    environment("test")
    service = _build_service()
    token = service.register("bypass@qq.com", "Passw0rd!234", INTEGRATION_TEST_FIXED_CODE)
    assert token
    # 注册确实落库（内存 store），不是空转
    assert service.store.get_user("bypass@qq.com") is not None


@pytest.mark.parametrize("env", ["production", "development"])
def test_non_test_env_rejects_fixed_code(environment, env):
    """硬要求：production（以及 development）必须拒绝固定码。"""
    environment(env)
    service = _build_service()
    with pytest.raises(ValueError):
        service.register("bypass@qq.com", "Passw0rd!234", INTEGRATION_TEST_FIXED_CODE)
    assert service.store.get_user("bypass@qq.com") is None


def test_non_test_env_accepts_real_code(environment):
    """对照：非 test 环境用真实验证码照常可注册（行为未被破坏）。"""
    environment("development")
    service = _build_service()
    real_code = service.send_code("real@qq.com", "register")
    token = service.register("real@qq.com", "Passw0rd!234", real_code)
    assert token


def test_bypass_logged_in_test_env(environment, caplog):
    environment("test")
    service = _build_service()
    with caplog.at_level(logging.INFO, logger="app.auth.service"):
        service.register("bypass@qq.com", "Passw0rd!234", INTEGRATION_TEST_FIXED_CODE)
    assert "集成测试旁路命中" in caplog.text


@pytest.mark.parametrize("env", ["production", "development"])
def test_bypass_log_absent_in_non_test_env(environment, caplog, env):
    """硬要求：生产/开发环境不得出现旁路日志。"""
    environment(env)
    service = _build_service()
    with caplog.at_level(logging.INFO, logger="app.auth.service"):
        with pytest.raises(ValueError):
            service.register("bypass@qq.com", "Passw0rd!234", INTEGRATION_TEST_FIXED_CODE)
    assert "集成测试旁路" not in caplog.text


def test_wrong_code_still_rejected_in_test_env(environment):
    """旁路只认固定码：test 环境下别的错误码仍被拒。"""
    environment("test")
    service = _build_service()
    with pytest.raises(ValueError):
        service.register("bypass@qq.com", "Passw0rd!234", "123456")


def test_reset_password_bypass(environment):
    """改密路径同样受旁路覆盖（同一校验入口）。"""
    environment("test")
    service = _build_service()
    service.register("reset@qq.com", "OldPass0rd!", INTEGRATION_TEST_FIXED_CODE)
    service.reset_password("reset@qq.com", "NewPass0rd!", INTEGRATION_TEST_FIXED_CODE)
    assert service.login("reset@qq.com", "NewPass0rd!")


def test_reset_password_rejects_fixed_code_in_production(environment):
    environment("production")
    service = _build_service()
    environment("test")
    service.register("reset@qq.com", "OldPass0rd!", INTEGRATION_TEST_FIXED_CODE)
    environment("production")
    with pytest.raises(ValueError):
        service.reset_password("reset@qq.com", "NewPass0rd!", INTEGRATION_TEST_FIXED_CODE)
