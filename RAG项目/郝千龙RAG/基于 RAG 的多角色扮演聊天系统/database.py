# -*- coding: utf-8 -*-  # 指定源文件编码为 utf-8，避免中文字符在读取时乱码
"""【数据库模型 · database.py】SQLAlchemy ORM 与建表：用户/角色/会话/消息/知识文档/知识块六张表（默认 SQLite，可换 MySQL）。"""  # 模块级文档字符串：中文名 + 文件名 + 一句话作用
from __future__ import annotations  # 启用 PEP 563 延迟注解求值，让类型注解字符串化，避免循环导入

from datetime import datetime  # 导入 datetime 类型，用于时间字段类型注解

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, create_engine, func  # 导入列类型、外键、引擎工厂及函数工具
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker  # 导入 2.0 声明式基类、类型注解映射工具和会话工厂

from config import DATABASE_URL  # 从配置文件读取数据库连接串（SQLite 或 MySQL 均可）
from logger import log  # 导入统一日志器，便于记录建表与初始化过程


class Base(DeclarativeBase):  # 继承 DeclarativeBase，定义所有 ORM 模型的公共基类
    """SQLAlchemy 2.0 声明式基类。"""  # 类文档字符串，说明基类用途

    pass  # 占位语句，Base 仅作为元数据容器，无额外字段


class User(Base):  # 用户表 ORM 模型，继承 Base 注册到元数据
    """用户表：用户名唯一，密码只存哈希。"""  # 文档字符串，强调安全策略：不存明文密码

    __tablename__ = "users"  # 指定数据库表名为 users
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)  # 主键 id，整型自增
    username: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)  # 用户名，最长 64 字符，唯一且非空
    password_hash: Mapped[str] = mapped_column(String(256), nullable=False)  # 密码哈希值，只存哈希避免泄露明文
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())  # 创建时间，由数据库在插入时填充当前时间


class Role(Base):  # 角色表 ORM 模型，用于持久化多角色提示词
    """角色表：与 roles.py 的提示词模板同步，实现多角色持久化。"""  # 文档字符串，说明与 roles.py 模板的同步关系

    __tablename__ = "roles"  # 指定数据库表名为 roles
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)  # 主键 id，整型自增
    code: Mapped[str] = mapped_column(String(32), unique=True, nullable=False)  # 角色代码（如 teacher），唯一非空，作为业务标识
    name: Mapped[str] = mapped_column(String(64), nullable=False)  # 角色展示名（如"老师"），非空
    domain: Mapped[str] = mapped_column(String(64), default="")  # 角色所属领域，默认空字符串
    description: Mapped[str] = mapped_column(Text, default="")  # 角色详细描述，长文本，默认空


class ChatSession(Base):  # 会话表 ORM 模型，一次多轮对话对应一行
    """会话表：一次多轮对话 = 一个 session，属于某个用户 + 某个角色。"""  # 文档字符串，说明会话与用户、角色的归属关系

    __tablename__ = "chat_sessions"  # 指定数据库表名为 chat_sessions
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)  # 主键 id，整型自增
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))  # 外键关联 users.id，标识会话属主
    role_code: Mapped[str] = mapped_column(String(32), nullable=False)  # 当前会话使用的角色代码，非空
    title: Mapped[str] = mapped_column(String(128), default="新对话")  # 会话标题，默认"新对话"，方便用户区分
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())  # 创建时间，数据库端默认当前时间
    updated_at: Mapped[datetime] = mapped_column(  # 更新时间，记录会话最近活动时间
        DateTime, server_default=func.now(), onupdate=func.now()  # 创建时填当前时间，每次 UPDATE 自动刷新
    )


class ChatMessage(Base):  # 消息表 ORM 模型，存储每条对话消息
    """消息表：会话内每条 user/assistant 消息全量落库（长期冷数据，可回溯）。"""  # 文档字符串，强调全量持久化用于历史回溯

    __tablename__ = "chat_messages"  # 指定数据库表名为 chat_messages
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)  # 主键 id，整型自增
    session_id: Mapped[int] = mapped_column(ForeignKey("chat_sessions.id"))  # 外键关联会话表，定位消息所属会话
    role: Mapped[str] = mapped_column(String(16), nullable=False)  # 消息角色（user/assistant），非空，区分对话双方
    content: Mapped[str] = mapped_column(Text, nullable=False)  # 消息正文，长文本非空，完整保存对话内容
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())  # 创建时间，用于按时间排序回溯


class KnowledgeDoc(Base):  # 知识文档表 ORM 模型，管理动态上传文档的元数据
    """知识文档表：记录动态上传文档的来源、摘要与分块数（知识库管理）。"""  # 文档字符串，说明用于知识库管理

    __tablename__ = "knowledge_docs"  # 指定数据库表名为 knowledge_docs
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)  # 主键 id，整型自增
    source: Mapped[str] = mapped_column(String(256), nullable=False)  # 文档来源（文件名/路径），非空，便于追溯
    summary: Mapped[str] = mapped_column(Text, default="")  # 文档摘要，长文本，默认空，供前端展示
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)  # 文档分块数量，默认 0，统计切片规模
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())  # 创建时间，数据库端默认当前时间
    updated_at: Mapped[datetime] = mapped_column(  # 更新时间，记录文档元数据最近修改时间
        DateTime, server_default=func.now(), onupdate=func.now()  # 创建时填当前时间，UPDATE 时自动刷新
    )


class KnowledgeChunk(Base):  # 知识块表 ORM 模型，存储角色数据集的每条原文
    """知识块表：五角色数据集（老师/医生/律师/心理医生/科学家）逐条入库的原文。  # 文档字符串起始：说明本表存储五角色数据集原文切片

    与 Milvus roleplay_kb 集合通过 chunk_uid 一一对应：  # 解释 SQL 表与向量库的对应关系
    SQL 存可事务查询的结构化原文，Milvus 存向量做语义检索，回表靠 chunk_uid。  # 双库分工：SQL 管结构化原文，Milvus 管向量，回表用 chunk_uid 关联
    """  # 多行文档字符串结束

    __tablename__ = "knowledge_chunks"  # 指定数据库表名为 knowledge_chunks
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)  # 主键 id，整型自增
    role_code: Mapped[str] = mapped_column(String(32), nullable=False, index=True)  # 所属角色代码，加索引加速按角色查询
    source: Mapped[str] = mapped_column(String(256), nullable=False)  # 来源文件名，非空，便于追溯数据出处
    question: Mapped[str] = mapped_column(Text, default="")  # 问题侧文本（英文/提问/来访者发言），默认空
    answer: Mapped[str] = mapped_column(Text, default="")    # 答案侧文本（中文/回复/咨询师发言），默认空
    content: Mapped[str] = mapped_column(Text, nullable=False)  # 拼接全文，用于向量化与 BM25 检索，非空
    chunk_uid: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)  # 幂等业务键，唯一标识每条 chunk，防重复入库
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())  # 创建时间，数据库端默认当前时间


# 数据库引擎 + 会话工厂（默认 SQLite，可切 MySQL）  # 模块级注释，说明引擎和会话工厂的用途与可切换性
engine = create_engine(DATABASE_URL, echo=False, future=True)  # 创建数据库引擎，echo=False 关闭 SQL 打印，future=True 启用 2.0 风格
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)  # 会话工厂，绑定引擎，关闭自动 flush 和自动提交，由业务显式控制


def init_db() -> None:  # 数据库初始化入口函数，无返回值
    """建表 + 写入种子数据（角色列表、guest 账号）。幂等，可重复调用。"""  # 文档字符串，说明建表+种子逻辑且可重复执行
    from auth import hash_password  # 延迟导入哈希函数，避免模块循环依赖
    from roles import ROLES  # 延迟导入角色模板字典，确保初始化时才加载

    Base.metadata.create_all(engine)  # 根据所有 ORM 模型定义创建对应表（已存在则跳过）
    with SessionLocal() as db:  # 打开一个数据库会话，with 块结束自动关闭连接
        # 首次启动：把 roles.py 的角色模板写入数据库  # 保留原有注释：说明种子角色写入逻辑
        if db.query(Role).count() == 0:  # 若 roles 表为空，判定为首次启动，需写入角色模板
            for item in ROLES.values():  # 遍历 roles.py 中每个角色模板字典
                db.add(  # 向会话添加新 Role 对象，等待提交
                    Role(  # 构造角色 ORM 实例
                        code=item["code"],  # 角色代码
                        name=item["name"],  # 角色展示名
                        domain=item["domain"],  # 角色领域
                        description=item["description"],  # 角色描述
                    )
                )
        # 默认 guest 账号（CLI 演示用）  # 保留原有注释：说明 guest 账号用途
        if db.query(User).filter_by(username="guest").first() is None:  # 若不存在 guest 用户则创建
            db.add(User(username="guest", password_hash=hash_password("guest123")))  # 新增 guest 用户，密码哈希后存库
        db.commit()  # 提交事务，将所有 add 操作持久化到数据库
    log.info("database ready: %s", DATABASE_URL)  # 记录日志：数据库已就绪，并打印连接串便于排查
