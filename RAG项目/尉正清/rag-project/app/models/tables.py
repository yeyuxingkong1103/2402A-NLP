# app/models/tables.py
"""MySQL ORM 表定义。

设计要点：
  - 角色人设/提示词模板存库，新增角色无需改代码
  - 会话与消息落库归档，Redis 只承担短期热记忆
  - Milvus 只存向量与最小检索载荷，业务字段以 MySQL 为准
"""
from sqlalchemy import (BigInteger, Boolean, Column, DateTime, ForeignKey,
                        Index, Integer, String, Text, func)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class User(Base):
    """用户信息"""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(64), nullable=False, unique=True, comment="登录名")
    password_hash = Column(String(128), nullable=False, comment="密码哈希")
    nickname = Column(String(64), default="", comment="昵称")
    status = Column(Boolean, default=True, nullable=False, comment="1正常 0禁用")
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    sessions = relationship("ChatSession", back_populates="user",
                            cascade="all, delete-orphan")

    def to_dict(self):
        return {"id": self.id, "username": self.username,
                "nickname": self.nickname, "status": bool(self.status)}


class Role(Base):
    """角色信息：人设与提示词模板由本表驱动，支持动态增删角色。"""
    __tablename__ = "roles"

    id = Column(Integer, primary_key=True, autoincrement=True)
    role_key = Column(String(64), nullable=False, unique=True, comment="角色标识，与 data/ 目录名一致")
    name = Column(String(64), nullable=False, comment="角色显示名")
    category = Column(String(32), default="", comment="领域分类")
    description = Column(String(512), default="", comment="角色简介")
    persona = Column(Text, comment="人设提示词")
    rules = Column(Text, comment="回答规则")
    greeting = Column(Text, comment="开场白")
    fallback = Column(Text, comment="知识库无命中时的兜底话术")
    disclaimer = Column(Text, comment="免责声明，自动附加到回答末尾")
    collection = Column(String(64), default="", comment="对应 Milvus 集合名")
    avatar = Column(String(255), default="")
    sort_order = Column(Integer, default=0)
    status = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, server_default=func.now())

    __table_args__ = (Index("idx_role_status", "status", "sort_order"),)

    def to_dict(self, with_prompt: bool = False):
        d = {
            "role_key": self.role_key, "name": self.name,
            "category": self.category, "description": self.description,
            "greeting": self.greeting, "avatar": self.avatar,
            "sort_order": self.sort_order, "status": bool(self.status),
        }
        if with_prompt:
            d.update({"persona": self.persona, "rules": self.rules,
                      "fallback": self.fallback, "disclaimer": self.disclaimer,
                      "collection": self.collection})
        return d


class ChatSession(Base):
    """会话：一个用户在一个角色下的一个对话线程。"""
    __tablename__ = "sessions"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    session_id = Column(String(64), nullable=False, unique=True, comment="对外暴露的会话ID")
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"),
                     nullable=False)
    role_key = Column(String(64), nullable=False)
    title = Column(String(128), default="", comment="会话标题，取首轮提问")
    turn_count = Column(Integer, default=0, comment="累计对话轮数")
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        Index("idx_session_user_role", "user_id", "role_key"),
    )

    user = relationship("User", back_populates="sessions")
    # 不与 Message 建 ORM relationship：两者以 session_id 字符串关联而非主键，
    # 且查询/级联删除都由 SessionService 显式完成，避免隐式 join 的开销。

    def to_dict(self):
        return {
            "session_id": self.session_id, "role_key": self.role_key,
            "title": self.title, "turn_count": self.turn_count,
            "created_at": self.created_at.strftime("%Y-%m-%d %H:%M:%S") if self.created_at else None,
            "updated_at": self.updated_at.strftime("%Y-%m-%d %H:%M:%S") if self.updated_at else None,
        }


class Message(Base):
    """消息归档：Redis 里过期的历史在这里可查。"""
    __tablename__ = "messages"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    session_id = Column(String(64), nullable=False)
    role = Column(String(16), nullable=False, comment="user / assistant")
    content = Column(Text, nullable=False)
    sources = Column(Text, comment="命中的知识来源，JSON 字符串")
    created_at = Column(DateTime, server_default=func.now())

    __table_args__ = (
        Index("idx_msg_session", "session_id", "id"),
    )

    def to_dict(self):
        return {
            "role": self.role, "content": self.content,
            "sources": self.sources,
            "created_at": self.created_at.strftime("%Y-%m-%d %H:%M:%S") if self.created_at else None,
        }


class LawIndex(Base):
    """法条结构化索引，支撑「元数据路召回」。

    向量检索对条文编号不敏感，而编号是精确信息。把法条号建成结构化索引放在
    MySQL 里，用户问「刑法第一百三十三条」时可以直接精确命中，
    不必指望语义相似度把对的那一条排上来。

    MySQL 存索引、Milvus 存向量与原文，各做擅长的事——这也是多路召回中
    MySQL 这一路的实际用途。
    """
    __tablename__ = "law_index"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    role_id = Column(String(64), nullable=False)
    law_name = Column(String(128), nullable=False, comment="法律全称")
    article = Column(String(64), nullable=False, comment="归一化条文号，如 第一百三十三条")
    chapter = Column(String(255), default="", comment="所属章节")
    doc_ref = Column(String(64), default="", comment="对应数据集记录 ID")
    source = Column(String(255), default="", comment="来源文件")

    __table_args__ = (
        Index("idx_law_article", "law_name", "article"),
        Index("idx_law_article_no", "article"),
    )

    def to_dict(self):
        return {"law_name": self.law_name, "article": self.article,
                "chapter": self.chapter, "source": self.source}
