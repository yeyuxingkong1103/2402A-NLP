"""认证用户主体的 MySQL 存储（批次5 认证落库）。

职责边界：
- 用户主体（users 表）读写走 MySQL：重启不丢用户，这是上线硬阻断的修复
- 验证码仍存进程内存（5 分钟 TTL 的短命数据，重启丢失只影响"重发一次"，
  不影响已注册用户的登录；后续如需可平移到 Redis，接口不变）
- 会话令牌机制不变：仍由 SessionStore 管 Redis（session:<token>）
- 密码哈希沿用 hash_password（PBKDF2-HMAC-SHA256，120k 轮，随机盐），
  本模块不做加解密，只存取哈希

与 app.chat.chat_store 的关系：
- chat_sessions.user_id 从本模块的 user.dataclass.user_key 取值落库；
  两边都是 32 位十六进制字符串，业务表不存自增 id（分库/换主键不牵动业务表）
"""

# 导入随机数生成（user_key 生成）
import secrets
# 导入时间比较（验证码过期判断）
import time

# 导入 SQLAlchemy 查询构造器
from sqlalchemy import select
# 导入 ORM 会话类型注解
from sqlalchemy.orm import Session, sessionmaker

# 导入认证侧的用户视图对象（存储无关的 dataclass）
from app.auth.service import User
# 导入 users 表 ORM 模型与统一时间函数
from app.db.sql_models import User as UserORM


class SqlAuthStore:
    """MySQL 版认证存储：用户读写落 users 表，验证码存内存。

    与 InMemoryAuthStore 接口一致，通过 AuthService 构造注入切换。
    """

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        """初始化落库存储。

        参数：
        - session_factory: SQLAlchemy 会话工厂（绑定 MySQL / 测试 SQLite）
        """
        self._session_factory = session_factory
        # 验证码存储（键为邮箱小写+用途，值为验证码+过期时间戳）；
        # 暴露为公开属性，与 InMemoryAuthStore.codes 保持同形，便于测试读取
        self.codes: dict[tuple[str, str], tuple[str, float]] = {}

    def get_user(self, email: str) -> User | None:
        """根据邮箱获取用户对象；不存在返回 None。"""
        with self._session_factory() as session:
            record = session.scalar(
                select(UserORM).where(UserORM.email == email.lower())
            )
            if record is None:
                return None
            # 拷贝为存储无关的 dataclass，脱离 ORM 会话后仍可用
            return User(
                user_key=record.user_key,
                email=record.email,
                password_hash=record.password_hash,
                is_admin=record.is_admin,
                is_active=record.is_active,
            )

    def save_user(self, user: User) -> None:
        """保存用户；user_key 缺失时生成（32 位十六进制），按 email upsert。

        说明：email 一律小写归一后再比对/落库，避免大小写重复注册。
        """
        if not user.user_key:
            user.user_key = secrets.token_hex(16)
        normalized_email = user.email.lower()
        user.email = normalized_email

        with self._session_factory() as session:
            record = session.scalar(
                select(UserORM).where(UserORM.email == normalized_email)
            )
            if record is None:
                # 新用户建档
                session.add(
                    UserORM(
                        user_key=user.user_key,
                        email=normalized_email,
                        password_hash=user.password_hash,
                        is_admin=user.is_admin,
                        is_active=user.is_active,
                    )
                )
            else:
                # 已有用户更新（改密 / 禁用启用 / 管理员标记）
                record.user_key = user.user_key
                record.password_hash = user.password_hash
                record.is_admin = user.is_admin
                record.is_active = user.is_active
            session.commit()

    def get_password_hash(self, email: str) -> str | None:
        """根据邮箱获取密码哈希；用户不存在返回 None。"""
        with self._session_factory() as session:
            record = session.scalar(
                select(UserORM.password_hash).where(UserORM.email == email.lower())
            )
            return record

    def save_code(self, email: str, purpose: str, code: str, expires_at: float) -> None:
        """保存验证码及其过期时间（内存存储，5 分钟 TTL 由调用方给定）。"""
        self.codes[(email.lower(), purpose)] = (code, expires_at)

    def consume_code(self, email: str, purpose: str, code: str) -> bool:
        """验证并消费验证码（一次性使用；过期或错误均返回 False）。"""
        key = (email.lower(), purpose)
        saved = self.codes.pop(key, None)
        return bool(saved and saved[0] == code and saved[1] > time.time())
