"""SMTPMailer 接入测试（批次 2 第二件）。

规则：
- 全部用替身（monkeypatch 假 smtplib），任何测试不得触发真实 SMTP 连接
- 凭据一律走 Settings（.env/环境变量），测试里用假值，不断言真实值
- production / development → SMTPMailer；测试服务需要内存邮件器时显式注入
"""
from dataclasses import dataclass, field

import pytest

from app.auth.mailer import (
    MailMessage,
    SMTPMailer,
    create_mailer,
)


# ==================== 替身 ====================


@dataclass
class StubSettings:
    """测试用配置替身，字段与 Settings 的 SMTP 相关项同名。"""

    environment: str = "development"
    smtp_host: str | None = "smtp.example.com"
    smtp_port: int = 465
    smtp_username: str | None = "bot@example.com"
    smtp_password: str | None = "fake-password"
    smtp_from_email: str | None = "bot@example.com"
    smtp_use_ssl: bool = True


class FakeSMTP:
    """记录调用的 smtplib 替身，绝不联网。"""

    calls: list[str] = []
    init_args: tuple = ()

    def __init__(self, host: str, port: int, timeout: float = 15.0) -> None:
        FakeSMTP.init_args = (host, port, timeout)
        FakeSMTP.calls.append("connect")

    def __enter__(self) -> "FakeSMTP":
        return self

    def __exit__(self, *exc_info: object) -> None:
        FakeSMTP.calls.append("close")

    def starttls(self) -> None:
        FakeSMTP.calls.append("starttls")

    def login(self, username: str, password: str) -> None:
        FakeSMTP.calls.append(f"login:{username}:{password}")

    def sendmail(self, from_addr: str, to_addrs: list, body: str) -> None:
        FakeSMTP.calls.append(f"sendmail:{from_addr}:{','.join(to_addrs)}")


@pytest.fixture()
def fake_smtp(monkeypatch: pytest.MonkeyPatch) -> type[FakeSMTP]:
    """把 smtplib 换成记录型替身（SMTP_SSL 与 SMTP 两个入口都换）。"""
    FakeSMTP.calls = []
    FakeSMTP.init_args = ()
    monkeypatch.setattr("app.auth.mailer.smtplib.SMTP_SSL", FakeSMTP)
    monkeypatch.setattr("app.auth.mailer.smtplib.SMTP", FakeSMTP)
    return FakeSMTP


# ==================== 按环境选择实现 ====================


def test_development_uses_smtp_mailer():
    """development 也必须使用 SMTPMailer，验证码发送到用户邮箱。"""
    mailer = create_mailer(StubSettings(environment="development"))
    assert isinstance(mailer, SMTPMailer)


def test_development_without_smtp_config_rejected():
    """development 缺少 SMTP 配置时直接报错，不允许假装发送成功。"""
    stub = StubSettings(environment="development", smtp_host=None)
    with pytest.raises(ValueError) as exc_info:
        create_mailer(stub)
    assert "SMTP_HOST" in str(exc_info.value)


def test_production_uses_smtp_mailer():
    """production → SMTPMailer（真发信）。"""
    mailer = create_mailer(StubSettings(environment="production"))
    assert isinstance(mailer, SMTPMailer)


def test_production_without_smtp_config_rejected():
    """production 但 SMTP 凭据缺失 → 直接报错，不允许静默降级。"""
    stub = StubSettings(environment="production", smtp_host=None)
    with pytest.raises(ValueError) as exc_info:
        create_mailer(stub)
    assert "SMTP_HOST" in str(exc_info.value)


# ==================== SMTPMailer 行为（替身，不联网） ====================


def test_smtp_mailer_send_ssl(fake_smtp: type[FakeSMTP]):
    """use_ssl=True → SMTP_SSL 直连：connect → login → sendmail。"""
    mailer = SMTPMailer(
        host="smtp.example.com",
        port=465,
        username="bot@example.com",
        password="fake-password",
        from_email="bot@example.com",
        use_ssl=True,
    )
    mailer.send(MailMessage(to="user@example.com", subject="验证码", body="code"))

    assert fake_smtp.init_args == ("smtp.example.com", 465, 15.0)
    assert any(call.startswith("login:bot@example.com:fake-password") for call in fake_smtp.calls)
    send_calls = [call for call in fake_smtp.calls if call.startswith("sendmail")]
    assert send_calls == ["sendmail:bot@example.com:user@example.com"]
    # SSL 直连不应走 starttls
    assert "starttls" not in fake_smtp.calls


def test_smtp_mailer_send_starttls(fake_smtp: type[FakeSMTP]):
    """use_ssl=False → 先明文连接再 STARTTLS 升级。"""
    mailer = SMTPMailer(
        host="smtp.example.com",
        port=587,
        username="bot@example.com",
        password="fake-password",
        from_email="bot@example.com",
        use_ssl=False,
    )
    mailer.send(MailMessage(to="user@example.com", subject="验证码", body="code"))

    assert fake_smtp.init_args == ("smtp.example.com", 587, 15.0)
    assert "starttls" in fake_smtp.calls
    send_calls = [call for call in fake_smtp.calls if call.startswith("sendmail")]
    assert send_calls == ["sendmail:bot@example.com:user@example.com"]


def test_smtp_mailer_rejects_blank_credentials():
    """凭据留空的 SMTPMailer 构造即报错（防呆，不等连上才炸）。"""
    with pytest.raises(ValueError):
        SMTPMailer(
            host="smtp.example.com",
            port=465,
            username="",
            password="x",
            from_email="bot@example.com",
        )


# ==================== 配置读取（键名契约） ====================


def test_settings_read_smtp_keys_from_environment(
    monkeypatch: pytest.MonkeyPatch,
):
    """Settings 必须从环境变量读 SMTP_USERNAME/PASSWORD/FROM_EMAIL/USE_SSL。"""
    from app.core.config import Settings

    monkeypatch.setenv("SMTP_USERNAME", "cfg-user@example.com")
    monkeypatch.setenv("SMTP_PASSWORD", "cfg-pass")
    monkeypatch.setenv("SMTP_FROM_EMAIL", "cfg-from@example.com")
    monkeypatch.setenv("SMTP_USE_SSL", "false")

    config = Settings.from_environment()
    assert config.smtp_username == "cfg-user@example.com"
    assert config.smtp_password == "cfg-pass"
    assert config.smtp_from_email == "cfg-from@example.com"
    assert config.smtp_use_ssl is False


def test_settings_smtp_use_ssl_defaults_true(monkeypatch: pytest.MonkeyPatch):
    """未设置 SMTP_USE_SSL → 默认 true（465 隐式 SSL）。"""
    from app.core.config import Settings

    monkeypatch.delenv("SMTP_USE_SSL", raising=False)
    config = Settings.from_environment()
    assert config.smtp_use_ssl is True
