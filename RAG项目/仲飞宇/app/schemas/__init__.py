"""Pydantic 请求 / 响应模型。

链路位置：FastAPI 用这些模型做两件事——请求体校验（不通过就在进路由函数之前返回 422，
所以越界输入根本到不了 pipeline），以及响应裁剪（response_model 决定前端能看到哪些字段，
多余的内部字段不会漏出去）。这里是接口契约的唯一定义处，改字段名等于改前端契约。

命名约定：XxxRequest 是入参，其余是出参。
"""
from __future__ import annotations

from pydantic import BaseModel, Field

# 角色 id 是标识符，会被拼进 Milvus 过滤表达式（见 core/store/milvus_store.py），
# 在此处就挡掉引号等能改写表达式的字符，非法输入直接 422 而不是走到 500。
ROLE_ID_PATTERN = r"^[A-Za-z0-9_.\-]{1,64}$"


class ChatRequest(BaseModel):
    # 上界是刻意的：提问与系统消息一起构成 prompt 的"固定开销"，没有上界时一条几万字的
    # 提问就能把 PROMPT_MAX_CHARS 吃光——历史被整段裁掉、prompt 仍然超窗，而日志只会说
    # 「系统消息 N 字 >= 上限」（见 prompt/templates.py），排查方向直接被带偏。
    question: str = Field(..., min_length=1, max_length=2000, description="用户问题")
    # 默认值指向 role_presets.ROLE_PRESETS 里的心理咨询师：不传 role_id 的请求落到它身上，
    # 所以那条预设不能删（删了默认请求就是 404）。改默认角色时，app/api/knowledge.py 的
    # Query 默认值要一起改，否则上传接口会往另一个角色上灌数据。
    role_id: str = Field(
        "psychologist", pattern=ROLE_ID_PATTERN,
        description="角色 id（字母/数字/下划线/点/连字符）",
    )
    # session_id 只做记忆隔离（记忆键是 f"{session_id}:{role_id}"），不做鉴权：
    # 谁猜到别人的 session_id 就能读到那段上下文，这是本项目的已知边界。
    # 限 128 是给 Redis / 内存两种后端的键长留余量。
    session_id: str = Field("default", max_length=128, description="会话 id（用于短期记忆隔离）")


class Source(BaseModel):
    text: str = ""
    title: str = ""
    source: str = ""
    score: float = 0.0


class ChatResponse(BaseModel):
    # 只服务非流式的 POST /chat。流式那条走 SSE，字段名与这里不同（sources / delta /
    # done.answer），两份契约要一起改（SSE 那份写在 docs/接口文档.md）。
    answer: str
    sources: list[Source] = []
    session_id: str = "default"


class DocumentInfo(BaseModel):
    id: int
    role_id: str
    source: str
    title: str
    chunk_count: int


class KnowledgeListResponse(BaseModel):
    documents: list[DocumentInfo]
    total: int


class RoleInfo(BaseModel):
    # 刻意不含 system_prompt：人设提示词是服务端资产，不回给前端，也免得前端拿它当可变状态。
    # 所以建完角色后前端拿不回提示词，要改只能整条重发（见 api/role.py 的 POST /role）。
    role_id: str
    name: str
    avatar: str = ""
    description: str = ""


class RoleCreateRequest(BaseModel):
    role_id: str = Field(..., pattern=ROLE_ID_PATTERN)
    name: str
    # 限 16 而不是 1：emoji 不都是单码点（🏋️、⚖️ 是「表情 + 变体选择符」两个码点），
    # 卡 1 会把这类写法挡在门外。上限与库里那列 String(16) 对齐。
    avatar: str = Field("", max_length=16, description="单个 emoji 头像，如 🥗")
    description: str = ""
    # 不设长度上界：它整段进 system 消息，写得过长会挤掉历史预算（templates.build_messages
    # 会记一条「预算被吃光」的告警），由使用方自己把握，不在这里硬卡。
    system_prompt: str


class HealthResponse(BaseModel):
    status: str
    components: dict[str, str]
    # 降级原因等补充说明，键名与 components 对应（如 {"memory": "Redis 不可达: ..."}）。
    # 只在真有话可说时出现；components 的值始终是 ok/degraded/unavailable 三档之一，
    # 原因串单独放这里，免得打断只读 components 做等值判断的调用方。
    details: dict[str, str] = Field(default_factory=dict)
