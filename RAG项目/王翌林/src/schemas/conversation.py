"""会话与消息 Schema（会话列表、消息列表、聊天请求/响应与引用来源）。

这里覆盖 RAG 陪伴系统的核心交互链路：
1) 前端创建会话 → ConversationCreateRequest；
2) 前端发一条消息 → ChatRequest；
3) 服务端返回回答 + 引用来源 → ChatResponse / ChatReference；
4) 历史会话与历史消息回显 → ConversationItem / MessageItem / MessageListResult。

请求模型负责“有效性校验”，响应模型负责“形状稳定”，
Model 里带 model_config = ConfigDict(from_attributes=True) 的，
表示可以由数据库 ORM 对象直接转换而来。
"""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class ConversationCreateRequest(BaseModel):
    """创建会话请求体：为一个具体的心理医生角色开一段新对话。"""
    # 角色 ID：必填，决定该会话使用哪套人格提示词与知识库
    persona_id: int
    # 会话标题：可选，不填则由服务端自动生成（如取首条消息摘要）。
    # 限长 255 与数据库 title 列宽一致，也避免超长标题撑破会话列表 UI
    title: Optional[str] = Field(default=None, max_length=255)


class ConversationItem(BaseModel):
    """会话列表中的单条记录（用于“我的会话”列表展示）。"""
    model_config = ConfigDict(from_attributes=True)  # 允许从 ORM 对象直接构造

    id: int
    # 会话归属用户，用于数据隔离；服务端会校验它等于当前登录用户
    user_id: int
    persona_id: int
    # 以下角色编码/名称是“冗余展示字段”：
    # 由后端 JOIN 角色表填入，让前端列表无需再发一次请求即可显示角色名
    persona_code: Optional[str] = None
    persona_name: Optional[str] = None
    title: Optional[str] = None
    # 会话状态：1=进行中，0=已归档/删除（软删除，保留历史数据）
    status: int = 1
    # 消息条数：列表页展示用，默认 0 表示尚无对话
    message_count: int = 0
    # 时间以字符串返回（由后端统一格式化），避免前后端时区解析歧义
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class MessageItem(BaseModel):
    """单条历史消息记录。"""
    model_config = ConfigDict(from_attributes=True)

    id: int
    conversation_id: int
    # 消息角色：只应是 "user"（用户）或 "assistant"（心理医生 AI）；
    # 这里不加枚举约束是为了兼容未来可能出现的 "system"/"tool" 等角色
    role: str
    content: str
    # 本条消息消耗的 token 数，可选：用于统计用量与成本，历史数据可能缺失
    tokens: Optional[int] = None
    # 该条回答引用的知识库片段（结构可变，故用 Any），
    # 用 Any 是为了兼容早期不同格式的历史数据
    refs: Optional[Any] = None
    created_at: Optional[str] = None


class MessageListResult(BaseModel):
    """消息列表结果：会话 ID + 总数 + 消息明细。"""
    conversation_id: int
    # 该会话消息总数，供前端判断是否还能继续上翻加载
    total: int
    items: List[MessageItem]


class ChatReference(BaseModel):
    """聊天回答的知识库引用来源（RAG 溯源信息）。

    全部字段可选：并非每轮回答都能检索到可用片段（如寒暄类问题），
    因此允许整条引用为空值上报。
    """
    # 命中的知识文档主键，前端可据此跳转到文档详情
    doc_id: Optional[int] = None
    # 命中的具体文本块主键，粒度为“段落级”，比文档级更精确
    chunk_id: Optional[int] = None
    # 来源标识（文件名或来源 URL），用于给用户展示“依据来自哪里”
    source: Optional[str] = None
    # 检索相似度分数，越高越相关；可用于前端展示置信度或做阈值提示
    score: Optional[float] = None


class ChatRequest(BaseModel):
    """聊天请求体：用户向某个心理医生角色发送一句话。"""
    persona_id: int
    # 用户消息内容：1~4000 字符。
    # 下限 1 防止空消息（空消息无法检索、也无意义）；
    # 上限 4000 是“单轮输入”的合理边界：再长会显著拉长上下文并挤占 RAG 检索片段预算
    message: str = Field(min_length=1, max_length=4000)
    # 会话 ID：可选。传了表示在这段已有会话中继续对话；
    # 不传则由服务端新建会话并返回新的 conversation_id
    conversation_id: Optional[int] = None
    # 是否流式返回（SSE 逐字输出）。默认 False 走一次性返回，实现更简单
    stream: bool = False


class ChatResponse(BaseModel):
    """聊天响应体（非流式场景）。"""
    conversation_id: int
    persona_id: int
    # 心理医生角色的完整回答文本
    answer: str
    # 回答依据的知识库引用列表，默认空列表
    references: List[ChatReference] = []
    # 本次交互 token 消耗量，默认 0，用于计费与用量统计
    tokens: int = 0
    # 结束原因："stop"=正常结束；其他值可表示超长截断等异常情况
    finish_reason: str = "stop"
    # 是否检测到危机信号（如自伤/自杀倾向）——这是心理陪伴产品的安全底线字段
    crisis_detected: bool = False
    # 危机干预提示语：当 crisis_detected 为 True 时，
    # 该字段承载面向用户的安抚话术与求助热线信息
    crisis_notice: Optional[str] = None