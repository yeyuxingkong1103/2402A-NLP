"""src/models/database.py —— 关系库的 SQLAlchemy 数据模型与连接。

在链路中的位置（最底层的数据定义）：
    被 src/offline/metadata_store.py（文档与分块）、src/api/deps.py 与各路由（用户/角色/会话/消息/反馈）引用

存储的内容 = 所有"结构化、需要按条件查询"的数据：
    user / role / session / message         —— 用户体系与对话记录
    knowledge_doc / knowledge_chunk         —— 知识库的文档登记与分块明细
    feedback                                —— 用户对回答的反馈（用于后续评估）

与 Milvus 的分工：向量存 Milvus（负责语义检索），
其余全部存这里（负责列表、排序、关联、审计）。

双数据库兼容（本项目的重要设计）：
    MYSQL_URL 指向 MySQL 就是生产形态；指向 sqlite:///... 时就是单机降级形态。
    同一个模型定义、同一套代码，换一个连接串就能从"需要 Docker 起 MySQL"
    变成"一个文件就是数据库" —— 这是本项目能在任何机器上跑通的关键之一。
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from configs.settings import get_settings


class Base(DeclarativeBase):
    """所有模型的基类（SQLAlchemy 2.0 的声明式写法）。

    继承 DeclarativeBase 后，子类通过 __tablename__ 与 mapped_column
    自动注册到 Base.metadata，init_db 里一句 create_all 就能建出全部表。
    """

    pass


class User(Base):
    """用户表。

    字段：
        id            自增主键
        username      用户名，唯一索引（防重名注册）
        password_hash 口令**哈希**值 —— 绝不存明文口令
        tenant_id     所属租户，索引加速按租户过滤
        created_at    注册时间
    """

    __tablename__ = "user"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(256))
    tenant_id: Mapped[str] = mapped_column(String(64), default="default", index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class Role(Base):
    """角色配置表。

    字段：
        id         自增主键
        role_id    业务角色标识（如 lawyer / psychologist），唯一索引
        tenant_id  所属租户
        config     整份角色配置的 JSON（persona / style / safety_notice / …）
        created_at / updated_at  创建与更新时间（updated_at 由 onupdate 自动维护）

    config 用 JSON 列而不是展开成一堆字段：
        角色卡的字段会随需求演化（今天加个"口头禅"、明天加个"知识库绑定"），
        用 JSON 存就不必每次都改表结构做迁移。
        代价是无法用 SQL 直接按角色属性查询 —— 本项目没有这种需求，取舍合适。

    id 与 role_id 两个标识的分工：
        id      数据库内部主键，用于外键关联
        role_id 业务标识，向量库里存的也是它（Milvus 无法关联关系库的自增主键）
        这种"双标识"在混合存储架构里很常见。
    """

    __tablename__ = "role"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    role_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    tenant_id: Mapped[str] = mapped_column(String(64), default="default", index=True)
    config: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ChatSession(Base):
    """会话表（一个用户在某个角色下的一段连续对话）。

    字段：
        id         自增主键
        user_id    外键 -> user.id
        role_id    业务角色标识（这里存字符串而不是外键，见下方说明）
        tenant_id  所属租户
        title      会话标题（由首条消息生成，便于在列表里辨认）
        created_at / updated_at  创建与最后活跃时间

    为什么 role_id 不做外键：
        角色既可以是数据库里定义的（Role 表），也可能是内置的、或来自配置文件。
        做成外键会把角色来源锁死成"必须在 Role 表里"，限制过强。
    """

    __tablename__ = "session"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("user.id"), index=True)
    role_id: Mapped[str] = mapped_column(String(64), index=True)
    tenant_id: Mapped[str] = mapped_column(String(64), default="default", index=True)
    title: Mapped[str] = mapped_column(String(256), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Message(Base):
    """消息表（对话的每一条发言）。

    字段：
        id         自增主键
        session_id 外键 -> session.id，索引加速按会话查历史
        user_id    外键 -> user.id
        role_id    业务角色标识
        speaker    发言方："user" 或 "assistant"
        content    消息正文（Text 类型，不限长度）
        created_at 发言时间

    role_id 和 user_id 在这里是冗余的（通过 session 就能查到）：
        有意为之的冗余 —— 按角色统计消息量、按用户审计时不必每次都 JOIN session 表。
        代价是需要保证冗余字段与 session 一致（由写入方保证）。
    """

    __tablename__ = "message"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("session.id"), index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("user.id"), index=True)
    role_id: Mapped[str] = mapped_column(String(64), index=True)
    speaker: Mapped[str] = mapped_column(String(32))
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class KnowledgeDoc(Base):
    """知识库文档登记表（一份文档一行）。

    字段：
        id            自增主键，Milvus 里的 doc_id 就用它（两个库靠它对齐）
        role_id       归属角色（同一份文件可以给不同角色各建一份）
        tenant_id     所属租户
        doc_source    文档来源（文件路径）
        status        构建状态：ready / building / failed（详见 metadata_store 的说明）
        metadata_json 附加元数据（解析器、分块数、分块策略等）
        created_at / updated_at

    这张表与 Milvus 记录靠 id 对应：
        Milvus 里每条向量记录都带 doc_id 字段，
        "删除这份文档"就是按 doc_id 删 Milvus 记录 + 删这张表的行（两边都要做）。
    """

    __tablename__ = "knowledge_doc"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    role_id: Mapped[str] = mapped_column(String(64), index=True)
    tenant_id: Mapped[str] = mapped_column(String(64), default="default", index=True)
    doc_source: Mapped[str] = mapped_column(String(512))
    status: Mapped[str] = mapped_column(String(32), default="ready")
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class KnowledgeChunk(Base):
    """知识库分块明细表（一个 chunk 一行）。

    字段：
        id            自增主键
        doc_id        外键 -> knowledge_doc.id，索引加速按文档取分块
        role_id       归属角色（冗余，便于直接按角色统计）
        tenant_id     所属租户
        content       分块正文（Text 不限长，实际入库时截断到 8192）
        summary       摘要（列表页展示用，避免拖回全文）
        parent_id     父块序号，可为空
        page          页码，默认 -1（表示"不适用"，如单页文档或记忆条目）
        metadata_json 附加元数据
        created_at

    parent_id 可为空且没有外键约束：
        它是"父子分块"结构里的逻辑序号（见 src/offline/chunkers.py 的 add_parent_child），
        不一定指向本表的主键，所以不能建成外键。
    """

    __tablename__ = "knowledge_chunk"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    doc_id: Mapped[int] = mapped_column(ForeignKey("knowledge_doc.id"), index=True)
    role_id: Mapped[str] = mapped_column(String(64), index=True)
    tenant_id: Mapped[str] = mapped_column(String(64), default="default", index=True)
    content: Mapped[str] = mapped_column(Text)
    summary: Mapped[str] = mapped_column(String(1024), default="")
    parent_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    page: Mapped[int] = mapped_column(Integer, default=-1)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class Feedback(Base):
    """用户反馈表（对某条回答的评价）。

    字段：
        id         自增主键
        user_id    外键 -> user.id
        session_id 外键 -> session.id（可空）
        message_id 外键 -> message.id（可空，指向被评价的那条回答）
        value      评分（Float，可支持点赞/点踩或 1-5 分等多种口径）
        comment    文字评论
        created_at

    session_id 与 message_id 允许为空：
        用户可能只想对整体体验给个反馈，而不针对某条具体回答。
    """

    __tablename__ = "feedback"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("user.id"), index=True)
    session_id: Mapped[int | None] = mapped_column(ForeignKey("session.id"), nullable=True)
    message_id: Mapped[int | None] = mapped_column(ForeignKey("message.id"), nullable=True)
    value: Mapped[float] = mapped_column(Float)
    comment: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


# 模块级引擎与会话工厂：全进程共用一个连接池，不能每次调用都新建
settings = get_settings()
# SQLite 特有处理：FastAPI 的多线程环境下，连接默认不允许跨线程复用，
# 关掉这个检查才能让同一个连接被不同请求线程使用（并发写安全由 SQLAlchemy 自己保证）。
# MySQL 不需要也不该带这个参数，所以按 url 前缀判断。
connect_args = {"check_same_thread": False} if settings.mysql_url.startswith("sqlite") else {}
# pool_pre_ping 是关键：连接池里的连接可能已被数据库端断开（超时/重启），
# 开启后每次取用前会先探活，避免"拿到一个死连接然后报错"
engine = create_engine(settings.mysql_url, connect_args=connect_args, pool_pre_ping=True)
# expire_on_commit=False：commit 后不让对象属性过期。
# 默认行为是 commit 后访问属性会重新查库，而本项目的常见用法是
# "提交后立刻用对象（如取 doc.id 去写 Milvus）"，开着会让每次访问都多一次查询
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


def init_db() -> None:
    """建出所有尚不存在的表（幂等，可反复调用）。

    create_all 只创建缺失的表，不会修改或删除已存在的表 ——
    所以它适合初始化，但不能用来做表结构变更（那需要迁移工具）。
    """
    Base.metadata.create_all(engine)


def db_session():
    """创建一个数据库会话。

    返回：
        Session 对象。

    注意这里**没有**做成上下文管理器：
        官方推荐的写法是 with SessionLocal() as session，
        但本项目的调用方（metadata_store 等）写的是
        `with db_session() as db:` —— 这依赖 Session 自身支持的
        上下文协议，退出时它会自动 close。
        这种写法不会自动 commit/rollback，所以每个写操作都要显式 commit，
        忘记 commit 会导致数据静默丢失。改动这部分代码时需特别注意。
    """
    return SessionLocal()
