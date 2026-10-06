"""用户、系统角色、用户偏好、日志相关表。

本模块负责“谁在使用系统”这一块的数据建模：
- User：账号主体；
- SysRole / UserSysRole：基于 RBAC 的角色权限（角色表 + 用户-角色关联表）；
- UserPersonaPreference：用户对心理医生角色的个性化偏好；
- AuditLog / LoginLog：操作审计与登录审计日志。
"""
import datetime as dt

# BigInteger：大整数（用于主键/外键，支持海量数据，避免 int 溢出）
# ForeignKey：声明外键关系；Index：为高频查询字段建索引；UniqueConstraint：联合唯一约束
from sqlalchemy import (BigInteger, DateTime, ForeignKey, Index, Integer, String,
                        UniqueConstraint, func)
from sqlalchemy.orm import Mapped, mapped_column

from src.models.base import Base


class User(Base):
    """用户表：系统注册账号的核心信息。

    一行代表一个用户；密码只保存哈希值，绝不存明文。
    """

    __tablename__ = "users"

    # 主键：BigInteger + 自增，保证单表可扩展到远大于 int 的数据量
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    # 登录名：唯一约束（unique=True）由数据库保证不重复，是登录时的主要查询条件
    username: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    # 密码哈希：只存不可逆哈希，即使库被拖走也无法直接还原明文密码（安全底线）
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    # 邮箱：可空，用于找回密码/通知；长度按常规邮箱上限设定
    email: Mapped[str] = mapped_column(String(128), nullable=True)
    # 手机号：可空，用于短信登录/提醒
    phone: Mapped[str] = mapped_column(String(32), nullable=True)
    # 昵称：展示用名称，允许为空
    nickname: Mapped[str] = mapped_column(String(64), nullable=True)
    # 头像：存储图片 URL（512 足够容纳带路径的地址），可空
    avatar: Mapped[str] = mapped_column(String(512), nullable=True)
    # 状态：1=启用、0=禁用；用默认值 default=1 让新用户默认可用，避免额外 UPDATE
    status: Mapped[int] = mapped_column(Integer, default=1)
    # 注册时间：由数据库写入默认值，应用无需干预
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, server_default=func.now())
    # 资料更新时间：每次修改用户信息时数据库自动刷新（onupdate）
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )
    # 最后登录时间：可空（从未登录过则为空），用于安全风控与活跃度统计
    last_login_at: Mapped[dt.datetime] = mapped_column(DateTime, nullable=True)

    # 表级约束/索引集合
    __table_args__ = (
        # 为 email、phone 建普通索引：登录/找回密码常按这两列查询，索引可显著提速
        Index("idx_users_email", "email"),
        Index("idx_users_phone", "phone"),
    )


class SysRole(Base):
    """系统角色表：RBAC 模型中的“角色”定义（如 admin、user）。

    角色本身只是一份字典数据，具体“谁拥有什么角色”存在 UserSysRole 关联表里。
    """

    __tablename__ = "sys_roles"

    # 主键
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    # 角色编码：唯一，程序内部判断权限时使用的稳定标识（如 "admin"），不随显示名变化
    role_code: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    # 角色显示名：给用户看的名称（如“管理员”）
    role_name: Mapped[str] = mapped_column(String(64), nullable=False)
    # 角色描述：可空，用于后台管理界面展示说明
    description: Mapped[str] = mapped_column(String(255), nullable=True)


class UserSysRole(Base):
    """用户-角色关联表：实现用户与角色的多对多关系。

    一个用户可拥有多个角色，一个角色也可授予多个用户，因此需要这张中间表。
    """

    __tablename__ = "user_sys_roles"

    # 主键
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    # 外键 -> users.id：关联到用户，不能为空
    user_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("users.id"), nullable=False)
    # 外键 -> sys_roles.id：关联到角色，不能为空
    role_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("sys_roles.id"), nullable=False)

    # 联合唯一约束：保证同一“用户-角色”组合只出现一次，防止重复授权产生脏数据
    __table_args__ = (UniqueConstraint("user_id", "role_id", name="uk_user_role"),)


class UserPersonaPreference(Base):
    """用户偏好表：记录某个用户选择/收藏的心理医生角色（persona）。

    同样是与 counselor_personas 的多对多关系，并额外用 is_default 标记默认角色。
    """

    __tablename__ = "user_persona_preferences"

    # 主键
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    # 外键 -> users.id：偏好所属用户
    user_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("users.id"), nullable=False)
    # 外键 -> counselor_personas.id：被偏好的心理医生角色
    persona_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("counselor_personas.id"), nullable=False
    )
    # 是否默认角色：1=是、0=否；默认 0，让用户显式指定默认项
    is_default: Mapped[int] = mapped_column(Integer, default=0)
    # 创建时间
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, server_default=func.now())

    # 联合唯一约束：同一用户对同一角色只保留一条偏好，避免重复记录
    __table_args__ = (UniqueConstraint("user_id", "persona_id", name="uk_user_persona"),)


class AuditLog(Base):
    """操作审计日志表：记录用户的关键操作，用于事后追责与安全排查。

    注意：user_id 这里故意不做外键，因为日志要长期保留、且可能记录已删除用户的操作。
    """

    __tablename__ = "audit_logs"

    # 主键
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    # 操作者用户 id：可空（系统自动操作无用户），不加外键以免日志随用户删除而失效
    user_id: Mapped[int] = mapped_column(BigInteger, nullable=True)
    # 操作动作标识：如 "login"、"delete_doc"，可空
    action: Mapped[str] = mapped_column(String(128), nullable=True)
    # 操作详情：保存上下文信息（较长，4000 字符），可空
    detail: Mapped[str] = mapped_column(String(4000), nullable=True)
    # 来源 IP：用于风控定位，可空
    ip: Mapped[str] = mapped_column(String(64), nullable=True)
    # 记录时间
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, server_default=func.now())

    # 按用户检索日志是最常见场景，故为 user_id 建索引
    __table_args__ = (Index("idx_audit_user", "user_id"),)


class LoginLog(Base):
    """登录日志表：专门记录每次登录事件，便于异常登录检测。

    与 AuditLog 分开建表，是为了让登录这一高频、结构固定的场景查询更快、字段更聚焦。
    """

    __tablename__ = "login_logs"

    # 主键
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    # 登录用户 id：可空（如登录失败可能无有效用户），同样不加外键以保留历史记录
    user_id: Mapped[int] = mapped_column(BigInteger, nullable=True)
    # 登录 IP：可空
    ip: Mapped[str] = mapped_column(String(64), nullable=True)
    # 浏览器/客户端标识（User-Agent）：512 长度以容纳完整 UA 字符串，可空
    user_agent: Mapped[str] = mapped_column(String(512), nullable=True)
    # 登录时间
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, server_default=func.now())

    # 常按用户查看登录历史，建索引加速
    __table_args__ = (Index("idx_login_user", "user_id"),)