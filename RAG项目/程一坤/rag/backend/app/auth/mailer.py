"""可替换的邮件发送协议与实现：内存版（开发联调）与 SMTP 版（生产真发信）。"""

import logging
import smtplib
from dataclasses import dataclass
from email.header import Header
from email.mime.text import MIMEText
from typing import Protocol

logger = logging.getLogger("app.auth.mailer")


@dataclass(frozen=True)
class MailMessage:
    to: str
    subject: str
    body: str


class Mailer(Protocol):
    def send(self, message: MailMessage) -> None: ...


class InMemoryMailer:
    """内存版：只记录不发真邮件。

    仅供测试通过依赖注入使用；运行时验证码必须由 SMTPMailer 发送到用户邮箱。
    """

    def __init__(self) -> None:
        self.messages: list[MailMessage] = []

    def send(self, message: MailMessage) -> None:
        self.messages.append(message)


class SMTPMailer:
    """SMTP 真发信（生产环境）。仅依赖标准库 smtplib，不引第三方。

    凭据（主机/端口/账号/密码/发件人）全部由调用方从 .env/环境变量
    传入，本类不做任何硬编码；日志只记收件人与结果，不记凭据。
    发送失败向上抛原始异常，由调用方决定重试与提示策略——
    静默吞掉会让用户以为验证码已发出，实际收不到。
    """

    def __init__(
        self,
        host: str,
        port: int,
        username: str,
        password: str,
        from_email: str,
        use_ssl: bool = True,
        timeout: float = 15.0,
    ) -> None:
        # 防呆：凭据不全直接构造失败，不等连接时才暴露
        missing = [
            name
            for name, value in (
                ("host", host),
                ("username", username),
                ("password", password),
                ("from_email", from_email),
            )
            if not value
        ]
        if missing:
            raise ValueError("SMTPMailer 缺少必需参数：" + "、".join(missing))
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.from_email = from_email
        self.use_ssl = use_ssl
        self.timeout = timeout

    def send(self, message: MailMessage) -> None:
        """发送一封纯文本邮件。SSL 直连（465）或 STARTTLS（587）二选一。"""
        mail = MIMEText(message.body, "plain", "utf-8")
        mail["Subject"] = Header(message.subject, "utf-8")
        mail["From"] = self.from_email
        mail["To"] = message.to

        if self.use_ssl:
            with smtplib.SMTP_SSL(
                self.host, self.port, timeout=self.timeout
            ) as server:
                server.login(self.username, self.password)
                server.sendmail(self.from_email, [message.to], mail.as_string())
        else:
            with smtplib.SMTP(self.host, self.port, timeout=self.timeout) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.sendmail(self.from_email, [message.to], mail.as_string())
        # 只记收件人，不记任何凭据
        logger.info("邮件已发送: to=%s", message.to)


def create_mailer(settings) -> Mailer:
    """按配置创建 SMTP 邮件发送器。

    所有运行环境都必须通过 SMTP 把验证码发送到用户邮箱；SMTP 配置缺失时
    直接报错，不允许降级成“发不出去还假装成功”。测试需要内存邮件器时，
    应由测试显式注入 `InMemoryMailer`。

    settings 只需提供 environment 与 smtp_* 字段（见 app.core.config.Settings）。
    """
    missing = [
        name
        for name, value in (
            ("SMTP_HOST", settings.smtp_host),
            ("SMTP_USERNAME", settings.smtp_username),
            ("SMTP_PASSWORD", settings.smtp_password),
            ("SMTP_FROM_EMAIL", settings.smtp_from_email),
        )
        if not value
    ]
    if missing:
        raise ValueError("SMTP 配置缺失：" + "、".join(missing))

    return SMTPMailer(
        host=settings.smtp_host,
        port=settings.smtp_port,
        username=settings.smtp_username,
        password=settings.smtp_password,
        from_email=settings.smtp_from_email,
        use_ssl=settings.smtp_use_ssl,
    )
