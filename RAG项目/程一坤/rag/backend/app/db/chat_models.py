"""聊天会话与消息的持久化模型（任务 5.4）。

实体定义按域分置；对外统一入口是 app/db/sql_models.py
（实体定义分散在 base/document_models/law_models/user_models/
 chat_models 各文件，改动表结构时务必确认涉及的所有域文件）。

建表约束：只允许 Base.metadata.create_all 补建缺失表，
禁止对已有表执行任何 drop / alter（数据安全红线）。
"""

# 导入时间类型（时间戳字段的类型注解使用）
from datetime import datetime

# 导入 SQLAlchemy 列类型与约束构造器
from sqlalchemy import DateTime, ForeignKey, Index, Integer, String, Text
# 导入 ORM 声明式映射基类与列定义工具
from sqlalchemy.orm import Mapped, mapped_column

# 复用全局声明基类（与 sql_models.py 同一个 metadata，create_all 时一并建表）
from app.db.base import Base
# 时间函数收敛到 sql_models 一份（本文件曾自带重复实现，已按冗余清理裁决删除）
from app.db.sql_models import utc_now


class ChatSession(Base):
    """聊天会话：一次多轮问答的容器，只能被归属人访问。

    命名遵循《目录与命名约定》：业务会话实体用 chat_session，
    避免与 SQLAlchemy 的数据库 Session（连接会话）混淆。
    """

    # 表名：chat_sessions（MySQL 实际表名）
    __tablename__ = "chat_sessions"
    # 会话列表按"归属人 + 最近活跃"查询，建联合索引
    __table_args__ = (
        Index("ix_chat_sessions_user_updated", "user_id", "updated_at"),
    )

    # 会话主键 ID（自增，仅内部关联使用，不对外暴露）
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # 对外会话 ID（调用方在 /chat/stream 里传的 session_id 就存这里）
    # 全局唯一：同一 session_key 只允许存在一个会话，重复创建走"查到即复用"
    session_key: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)

    # 归属用户标识（VARCHAR(64) 不加外键），存认证侧的 user_key（32 位十六进制）
    # 切换说明（批次5 认证落库）：用户主体已从 InMemoryAuthStore 迁到 MySQL users 表，
    # 认证服务签发令牌时绑定 user_key，本列自切换起一律存 user_key；
    # 切换时 users 0 行、存量会话均为旧内存认证遗留的孤儿会话（user_id 指向
    # 已不存在的用户，归属校验必然 404，无数据风险），故未做数据迁移。
    # 不加外键：业务表只依赖 user_key 字符串，未来换主键/分库不牵动本表；
    # 且孤儿 user_key 不阻塞写入（外键会在无对应用户行时立即违规）
    user_id: Mapped[str] = mapped_column(String(64), nullable=False)

    # 助手角色 ID（首期只有一个预设法律知识助手，固定 "legal-assistant"）
    character_id: Mapped[str] = mapped_column(String(64), nullable=False)

    # 会话标题（生成规则：首条提问清洗后取前 20 字；显式创建接口可指定）
    title: Mapped[str] = mapped_column(String(256), nullable=False)

    # 创建时间（UTC，写库时自动填充）
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )

    # 最近活跃时间（UTC，任何一次问答落库都会刷新，用于会话列表排序）
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False
    )


class ChatMessage(Base):
    """聊天消息：会话内的单条发言，user 与 assistant 成对出现。

    citations / model 仅 assistant 消息使用；user 消息两者为 NULL。
    """

    # 表名：chat_messages（MySQL 实际表名）
    __tablename__ = "chat_messages"
    # 拉取历史按"会话 + 消息主键正序"，建联合索引
    __table_args__ = (
        Index("ix_chat_messages_session_id", "session_id", "id"),
    )

    # 消息主键 ID（自增；同会话内 id 递增即时间正序）
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # 所属会话 ID（外键保留：会话表是本模块自建的真实 MySQL 表，无 FK 违规风险）
    # 会话被删时消息级联语义由应用层处理（本轮未做删除接口，暂不设 ondelete）
    session_id: Mapped[int] = mapped_column(
        ForeignKey("chat_sessions.id"), nullable=False
    )

    # 发言角色（"user" = 提问方，"assistant" = 助手回答；与 query_rewrite 的期望结构一致）
    role: Mapped[str] = mapped_column(String(16), nullable=False)

    # 对外消息 ID（与 SSE message_start 事件携带的值完全一致，客户端据此把
    # 流式回答与历史消息关联起来；仅 assistant 消息写入，
    # user 消息与本列添加前的历史行为 NULL）
    message_id: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # 消息正文（提问原文 / 回答最终文本，含护栏处理后的免责声明）
    content: Mapped[str] = mapped_column(Text, nullable=False)

    # 引用法源列表（仅 assistant；ChatResult.sources 的 JSON 序列化，无引用时为 NULL）
    citations: Mapped[str | None] = mapped_column(Text, nullable=True)

    # 生成模型名（仅 assistant；LLM 客户端当前未暴露模型信息时为 NULL，禁止伪造）
    model: Mapped[str | None] = mapped_column(String(128), nullable=True)

    # 创建时间（UTC，写库时自动填充）
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
