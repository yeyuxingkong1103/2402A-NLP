"""用户域实体定义：登录用户（认证与长期记忆开关）。

实体定义按域分置；对外统一入口是 app/db/sql_models.py
（调用方一律从 sql_models 导入，不要直接 import 本文件，
否则实体定义分散后容易出现两套 import 并存的维护盲区）。
"""

from datetime import datetime

from sqlalchemy import Boolean, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, utc_now


class User(Base):
    """登录用户；只保存密码哈希，不保存明文密码。

    批次5 认证落库对齐后的结构（MySQL 已用显式迁移脚本同步）：
    - 对外稳定标识 user_key（32 位十六进制随机串），业务表
      （chat_sessions.user_id 等）一律存 user_key，不存自增 id——
      以后换主键或分库不会牵动业务表，也避免对外暴露用户数量
    - username 列已随迁移删除（认证代码从未使用，按邮箱前缀生成的
      用户名与登录体系无关）
    """

    __tablename__ = "users"

    # 用户主键 ID（自增，仅内部使用，不对外暴露）
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    # 对外稳定标识（32 位十六进制，注册时生成，全局唯一）
    user_key: Mapped[str] = mapped_column(String(32), unique=True, nullable=False)
    # 用户邮箱（唯一，登录主体）
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    # 密码哈希（PBKDF2-HMAC-SHA256，盐+摘要 base64；不存储明文密码）
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    # 是否管理员（阶段6 审核发布启用；注册默认 False）
    is_admin: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    # 账户是否激活（False 时登录被拒，对外不暴露禁用状态）
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    # 长期记忆开关（批次 14，接口 8.3）：False 时既不写入也不读取该用户的长期记忆
    long_term_memory_enabled: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True
    )
    # 创建时间
    created_at: Mapped[datetime] = mapped_column(default=utc_now, nullable=False)
    # 更新时间
    updated_at: Mapped[datetime] = mapped_column(default=utc_now, onupdate=utc_now, nullable=False)
