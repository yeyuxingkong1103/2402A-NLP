"""会话与消息表。

对话侧的表结构：
- Conversation：一次会谈（会话）的元信息，绑定用户与心理医生角色；
- Message：会话中的一条条消息（用户提问 / AI 回复），是聊天记录的最小单元。
"""
import datetime as dt
# Optional/Dict/Any：用于标注 refs（引用的知识片段）这类 JSON 字段
from typing import Any, Dict, Optional

from sqlalchemy import JSON, BigInteger, DateTime, ForeignKey, Index, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from src.models.base import Base


class Conversation(Base):
    """会话表：一次持续的心理咨询对话。

    一个用户可与不同人设分别开启多个会话；message_count 冗余记录消息条数。
    """

    __tablename__ = "conversations"

    # 主键
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    # 外键 -> users.id：会话归属的用户，不可空
    user_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("users.id"), nullable=False)
    # 外键 -> counselor_personas.id：本次会话使用的心理医生角色，不可空
    persona_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("counselor_personas.id"), nullable=False
    )
    # 会话标题：可按首条消息自动生成或用户自定义，可空
    title: Mapped[str] = mapped_column(String(255), nullable=True)
    # 状态：1=进行中、0=已结束/归档；用于会话列表过滤
    status: Mapped[int] = mapped_column(Integer, default=1)
    # 消息数量：默认 0，冗余计数，避免每次都 count messages 表即可展示条数
    message_count: Mapped[int] = mapped_column(Integer, default=0)
    # 创建时间
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, server_default=func.now())
    # 更新时间：有新消息时刷新，可用于按“最近活跃”排序会话列表
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )

    # 按用户查会话、按人设查会话都是列表页高频操作，各建索引加速
    __table_args__ = (
        Index("idx_conv_user", "user_id"),
        Index("idx_conv_persona", "persona_id"),
    )


class Message(Base):
    """消息表：会话中的单条对话内容（一问一答都存这里）。

    role 区分消息来源，content 为正文，refs 保存 RAG 检索到的知识出处以便溯源。
    """

    __tablename__ = "messages"

    # 主键
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    # 外键 -> conversations.id：所属会话，删除会话时可据此清理消息
    conversation_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("conversations.id"), nullable=False
    )
    # 角色标识：user=用户、assistant=AI（沿用 LLM 对话协议），决定消息如何渲染
    role: Mapped[str] = mapped_column(String(32), nullable=False)
    # 消息正文：Text 类型，用户长文本或 AI 长回答都能容纳，不可空
    content: Mapped[str] = mapped_column(Text, nullable=False)
    # token 数：可空，用于统计用量与上下文窗口控制（成本核算）
    tokens: Mapped[int] = mapped_column(Integer, nullable=True)
    # 引用来源：JSON 数组，记录本条回答引用了哪些知识分块（实现“有据可查”），可空；
    # 用 JSON 存储是因为引用数量与结构不固定，避免为它单开关联表
    refs: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    # 发送时间：展示消息顺序与时间线的依据
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, server_default=func.now())

    # 按会话拉取全部消息是最频繁的查询，建索引保证翻页与加载速度
    __table_args__ = (Index("idx_msg_conv", "conversation_id"),)