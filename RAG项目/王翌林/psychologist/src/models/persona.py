"""心理医生角色表。

本表定义系统中的“AI 心理医生人设”（persona）：每种疗法流派/性格风格对应一条记录，
包含提示词、问候语、可用知识范围、安全边界以及模型参数等，是 RAG 对话的“人格配置源”。
"""
import datetime as dt
# Any/Dict/Optional：用于标注 model_params 这类 JSON 字段的类型
from typing import Any, Dict, Optional

# JSON：以 JSON 列存储灵活的结构化配置；Text：不限长文本（提示词/边界可能很长）
from sqlalchemy import JSON, BigInteger, DateTime, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from src.models.base import Base


class CounselorPersona(Base):
    """心理医生人设表：一个角色 = 一套对话行为与知识范围的配置。

    对话时，系统会取出这里的 system_prompt、问候语、安全边界等，拼装成 LLM 的上下文。
    """

    __tablename__ = "counselor_personas"

    # 主键
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    # 人设编码：唯一，程序内部稳定引用某个角色的标识（如 "cbt_coach"），不随显示名变化
    persona_code: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    # 角色名称：展示给用户的名字（如“温暖倾听者”）
    name: Mapped[str] = mapped_column(String(64), nullable=False)
    # 头衔/称谓：如“心理咨询师”，可空
    title: Mapped[str] = mapped_column(String(128), nullable=True)
    # 疗法类型：如 CBT、精神分析，可空，用于分类筛选
    therapy_type: Mapped[str] = mapped_column(String(64), nullable=True)
    # 沟通风格：自然语言描述语气风格，可空
    style: Mapped[str] = mapped_column(String(255), nullable=True)
    # 常用方法：该角色擅长的干预技术，可空
    methods: Mapped[str] = mapped_column(String(255), nullable=True)
    # 开场问候语：进入会话时展示的首句话，Text 类型以容纳较长文案
    greeting: Mapped[str] = mapped_column(Text, nullable=True)
    # 系统提示词：定义角色行为的核心指令，对话生成必用，故不可空
    system_prompt: Mapped[str] = mapped_column(Text, nullable=False)
    # 知识范围：限定该角色检索哪些知识库文档，可空（空表示不限制）
    knowledge_scope: Mapped[str] = mapped_column(String(255), nullable=True)
    # 头像图片地址，可空
    avatar: Mapped[str] = mapped_column(String(512), nullable=True)
    # 安全边界：约束角色不得越界的内容（如禁止诊断、危机转介），Text 类型，可空
    safety_boundary: Mapped[str] = mapped_column(Text, nullable=True)
    # 模型参数：JSON 存储 temperature、max_tokens 等可变键值对，可空；
    # 用 JSON 而非固定列，是为了不同模型/角色可自由扩展参数而无需改表结构
    model_params: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    # 状态：1=启用、0=停用；停用后用户不可再选择该角色
    status: Mapped[int] = mapped_column(Integer, default=1)
    # 创建时间
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, server_default=func.now())
    # 更新时间：配置修改后自动刷新
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )