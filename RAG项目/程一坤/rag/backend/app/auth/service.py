"""认证核心服务：注册 / 登录 / 改密 / 验证码 / 令牌签发。

用户主体存储（批次5 起分两条路径）：
- 生产：SqlAuthStore（MySQL users 表，重启不丢用户）
- 测试：InMemoryAuthStore（进程内存，仅测试用）
两者对 AuthService 暴露同一接口（get_user / save_user / get_password_hash /
save_code / consume_code），通过构造注入切换。
"""

import base64
import hashlib
import hmac
import logging
import secrets
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from app.auth.mailer import InMemoryMailer, MailMessage, Mailer
from app.auth.session_store import SessionStore
from app.core.config import settings

if TYPE_CHECKING:
    # 仅类型注解使用，避免与 sql_store 循环导入
    from app.auth.sql_store import SqlAuthStore

logger = logging.getLogger("app.auth.service")

# 集成测试旁路固定码（批次 38）。
# 仅用于集成测试，生产禁止启用：只有 ENVIRONMENT=test 时才接受该固定码；
# development / production 一律走 store 里的真实验证码校验（不可达此分支）。
INTEGRATION_TEST_FIXED_CODE = "000000"


def is_integration_test_bypass(code: str) -> bool:
    """判断是否命中集成测试旁路（仅 ENVIRONMENT=test 生效）。

    环境判断放在最前面，保证 development / production 永远不会因固定码放行。
    """
    return settings.environment == "test" and code == INTEGRATION_TEST_FIXED_CODE


@dataclass
class User:
    """认证模块的用户对象（存储无关的轻量视图）。

    批次5 认证落库后，生产路径由 SqlAuthStore 读写 MySQL（users 表），
    测试路径仍可用 InMemoryAuthStore；两者对 AuthService 暴露同一接口。
    user_key 是对外稳定标识（32 位十六进制），业务表一律存它而非自增 id。
    """

    # 对外稳定标识（注册时生成；业务表如 chat_sessions.user_id 存此值）
    user_key: str = ""
    email: str = ""
    password_hash: str = ""
    is_admin: bool = False
    is_active: bool = True


@dataclass
class StoredUser:
    """内存存储的用户信息（仅测试用）。"""

    user: User


class InMemoryAuthStore:
    """内存认证存储（仅测试用，生产环境应使用 Redis + MySQL）。"""

    def __init__(self) -> None:
        # 存储用户对象（键为邮箱小写）
        self.users: dict[str, StoredUser] = {}
        # 存储验证码（键为邮箱+用途，值为验证码+过期时间）
        self.codes: dict[tuple[str, str], tuple[str, float]] = {}

    def get_user(self, email: str) -> User | None:
        """根据邮箱获取用户对象。"""
        stored = self.users.get(email.lower())
        return stored.user if stored else None

    def save_user(self, user: User) -> None:
        """保存用户对象，并生成稳定 user_key（若未设置）。"""
        if not user.user_key:
            user.user_key = secrets.token_hex(16)
        self.users[user.email.lower()] = StoredUser(user)

    def get_password_hash(self, email: str) -> str | None:
        """根据邮箱获取密码哈希。"""
        stored = self.users.get(email.lower())
        return stored.user.password_hash if stored else None

    def save_code(self, email: str, purpose: str, code: str, expires_at: float) -> None:
        """保存验证码及其过期时间。"""
        self.codes[(email.lower(), purpose)] = (code, expires_at)

    def consume_code(self, email: str, purpose: str, code: str) -> bool:
        """验证并消费验证码（一次性使用）。"""
        key = (email.lower(), purpose)
        saved = self.codes.pop(key, None)
        return bool(saved and saved[0] == code and saved[1] > time.time())


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 120_000)
    return base64.urlsafe_b64encode(salt + digest).decode()


def verify_password(password: str, encoded: str) -> bool:
    try:
        raw = base64.urlsafe_b64decode(encoded.encode())
        salt, expected = raw[:16], raw[16:]
        actual = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 120_000)
        return hmac.compare_digest(actual, expected)
    except (ValueError, TypeError):
        return False


class AuthService:
    """认证服务，处理注册、登录、验证码和令牌签发。"""

    def __init__(
        self,
        store: "InMemoryAuthStore | SqlAuthStore | None" = None,
        mailer: Mailer | None = None,
        session_store: SessionStore | None = None,
    ) -> None:
        # 用户和验证码存储
        self.store = store or InMemoryAuthStore()
        # 邮件发送服务
        self.mailer = mailer or InMemoryMailer()
        # 会话存储
        if session_store is None:
            raise ValueError("session_store 必须提供，不能为 None")
        self.session_store = session_store

    def _consume_code(self, email: str, purpose: str, code: str) -> bool:
        """校验并一次性消费验证码（所有验证码校验的唯一入口）。

        集成测试旁路（仅用于集成测试，生产禁止启用）：
        ENVIRONMENT=test 时固定码 000000 直接放行——集成测试起独立进程连真
        MySQL/Redis，读不到进程内的真实验证码，也无法走 SMTP 收信。
        该分支的进入条件先判 ENVIRONMENT，development / production 不可达，
        因此下面的 INFO 日志也只可能出现在 test 环境。
        """
        if is_integration_test_bypass(code):
            logger.info(
                "集成测试旁路命中：固定验证码放行（environment=test, purpose=%s）",
                purpose,
            )
            return True
        return self.store.consume_code(email, purpose, code)

    def send_code(self, email: str, purpose: str) -> str:
        """生成并发送验证码到用户邮箱。"""
        # 生成 6 位随机数字验证码
        code = f"{secrets.randbelow(1_000_000):06d}"
        # 保存验证码，5 分钟有效
        self.store.save_code(email, purpose, code, time.time() + 300)
        # 发送邮件
        self.mailer.send(
            MailMessage(email, "Legal RAG 验证码", f"验证码：{code}，5 分钟内有效。")
        )
        return code

    def register(self, email: str, password: str, code: str) -> str:
        """验证码校验通过后注册新用户并返回访问令牌。"""
        # 先查重：邮箱已注册直接拒绝（错误信息明确，避免与验证码错误混淆）
        if self.store.get_user(email):
            raise ValueError("该邮箱已注册")
        # 验证码校验（一次性消费）
        if not self._consume_code(email, "register", code):
            raise ValueError("验证码无效或已过期")
        # 创建新用户对象（user_key 由 save_user 生成）
        user = User(
            email=email,
            password_hash=hash_password(password),
        )
        # 保存用户（生成 user_key）
        self.store.save_user(user)
        # 签发会话令牌（绑定 user_key，业务表统一用此标识）
        if not user.user_key:
            raise RuntimeError("用户标识生成失败")
        return self.session_store.create_session_token(
            user_id=user.user_key, is_admin=user.is_admin
        )

    def login(self, email: str, password: str) -> str:
        """验证邮箱和密码后返回访问令牌。"""
        # 获取用户的密码哈希
        password_hash = self.store.get_password_hash(email)
        # 获取用户对象
        user = self.store.get_user(email)
        # 验证用户是否存在、是否激活、密码是否正确（禁用与密码错误同一句话，不暴露状态）
        if (
            not user
            or not password_hash
            or not user.is_active
            or not verify_password(password, password_hash)
        ):
            raise PermissionError("邮箱或密码错误")
        # 签发会话令牌（绑定 user_key）
        if not user.user_key:
            raise RuntimeError("用户标识不存在")
        return self.session_store.create_session_token(
            user_id=user.user_key, is_admin=user.is_admin
        )

    def reset_password(self, email: str, password: str, code: str) -> None:
        """验证码校验通过后重置用户密码。"""
        # 获取用户对象
        user = self.store.get_user(email)
        # 验证用户是否存在且验证码是否有效
        if not user or not self._consume_code(email, "reset", code):
            raise ValueError("验证码无效或已过期")
        # 更新密码哈希
        user.password_hash = hash_password(password)
        # 保存用户
        self.store.save_user(user)
        # 撤销该用户的所有旧会话（改密码后旧令牌必须立即失效）
        if not user.user_key:
            raise RuntimeError("用户标识不存在")
        self.session_store.revoke_all_sessions(user.user_key)

    def logout(self, token: str) -> None:
        """注销单个会话。"""
        self.session_store.revoke_session(token)
