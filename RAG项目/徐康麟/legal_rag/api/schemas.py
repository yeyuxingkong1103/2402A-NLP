# -*- coding: utf-8 -*-
"""HTTP 请求 / 响应模型（Pydantic）—— 从 ``api/app.py`` 拆出的第一个模块。

为什么要单独一个文件
--------------------
这些模型原先堆在 ``api/app.py``（1,882 行）里，与路由、装配逻辑混在一起。
它们**没有**任何闭包依赖、也不调用业务代码，是纯声明 ⇒ 拆出来零风险，
而且能被多个路由模块与服务层共享。

⚠️ 本模块是**对外契约**的一部分：字段名、默认值、约束（``min_length`` / ``pattern``）
一旦改动就是接口变更。特别是下面三处"用默认值代替必填"的刻意选择，改动前请先读注释：
* :class:`ResetRequest` —— 缺字段必须由业务层判成 **400**，不能用 pydantic 的必填（那会变 422）；
* :class:`PrefsUpdate` —— ``theme=None`` 表示"显式跟随系统"，与"本次不传"不同
  （调用方用 ``model_fields_set`` 区分）；
* :class:`SessionRenameRequest` —— 标题**不做 trim/截断**（原样落库），空串是合法值。
"""
from __future__ import annotations

from pydantic import BaseModel, Field

__all__ = [
    "ChatRequest", "Citation", "ChatResponse",
    "SessionCreateRequest", "SessionResponse", "SessionRenameRequest",
    "SessionMessage", "SessionMessagesResponse", "RoleResponse",
    "PrefsUpdate", "PrefsResponse", "IngestRequest",
    "RegisterRequest", "LoginRequest", "ResetRequest",
]


# ---------------- 对话 ----------------

class ChatRequest(BaseModel):
    user_id: str = Field(..., min_length=1, description="用户 ID")
    role_id: str = Field(..., min_length=1, description="角色 ID 或中文名")
    session_id: str = Field(..., min_length=1, description="会话 ID")
    message: str = Field(..., min_length=1, description="用户提问")
    top_k: int | None = Field(None, ge=1, le=50, description="最终保留的知识条数")
    stream: bool = Field(False, description="是否流式返回（text/plain 分块）")
    #: 流式协议：``text``（默认，裸文本分块，行为与既往完全一致）；
    #: ``sse`` 才会在流末补一帧引用（网页聊天需要引用，又不想为此放弃流式）。
    stream_mode: str = Field("text", pattern="^(text|sse)$",
                             description="流式协议：text=裸文本分块（默认）/ sse=带引用帧")


class Citation(BaseModel):
    source: str
    chunk_id: str
    score: float
    snippet: str
    # 条号（如「第五百零六条」）：引用落条用，界面展示它而**不再展示 score**
    # （score 是向量+关键词两路归一化后的融合分，上限 2.0、跨条不可比）。
    article: str = ""


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation]
    session_id: str
    role_id: str
    provider: str
    #: 本轮是否走了兜底（t120 F1）：为 True 时 provider 通常是 ``mock`` ——
    #: 前端据此显示"当前是演示回答/服务降级"的提示，并可提供重试入口。
    degraded: bool = False
    #: 兜底原因（可定位口径，见 ``legal_rag.generate.llm_base.failure_reason``）：
    #: missing_api_key / connect_failed / http_5xx / http_4xx / timeout / …
    degraded_reason: str = ""


# ---------------- 会话 ----------------

class SessionCreateRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    role_id: str = Field(..., min_length=1)
    session_id: str | None = Field(None, description="留空则自动生成")
    title: str = ""


class SessionResponse(BaseModel):
    session_id: str
    user_id: str
    role_id: str
    title: str = ""


class SessionRenameRequest(BaseModel):
    """重命名会话的请求体（`PATCH /sessions/{session_id}`）。"""
    user_id: str = Field(..., min_length=1,
                         description="用户 ID；AUTH_REQUIRED=true 时必须等于登录态")
    #: 长度上限与 ``sessions.title VARCHAR(255)`` 对齐；**不做 trim/截断**（原样落库，
    #: 避免"用户看到的和存的不一样"）；空串是合法值（列表里回落到显示 session_id）。
    title: str = Field("", max_length=255, description="新标题（≤255，原样保存）")


class SessionMessage(BaseModel):
    role: str          # user | assistant | system
    content: str
    created_at: float = 0.0
    #: 助手消息当时的引用来源（与 `/chat` 的 citations 同源、同字段）。
    #: **旧消息没有这个字段时按"无引用来源"处理**（空列表），不报错。
    citations: list[dict] = Field(default_factory=list)
    #: B-9：消息的稳定 ID（`AC-ST-4` 要求历史逐条含 message_id，便于与关系库逐条对齐）。
    message_id: str = ""


class SessionMessagesResponse(BaseModel):
    """`GET /sessions/{session_id}/messages` 的响应体。

    ``role_id`` 由**服务端**从 `sessions` 表读出（短期记忆的隔离键包含 role_id，
    不能让客户端自己拼），因此它出现在响应里供前端核对。
    """
    session_id: str
    user_id: str
    role_id: str
    title: str = ""
    messages: list[SessionMessage] = Field(default_factory=list)


# ---------------- 角色 / 偏好 ----------------

class RoleResponse(BaseModel):
    role_id: str
    name: str
    domain: str
    persona: str


class PrefsUpdate(BaseModel):
    """账号偏好的**写入**载荷（B-8 / r17 §21）。

    只有两个业务字段：`theme` 与 `role_id`（`AC-PR-1` 的字段白名单）。
    `theme=None` 表示**显式**选择"跟随系统"（落库为 NULL），与"本次不传 theme"不同 ——
    调用方用 `model_fields_set` 区分两者。

    ``user_id`` 只是为了兼容旧前端可能顺手带上的字段：**服务端一律忽略**，
    身份只从登录态派生（`AC-PR-5④`）。
    """

    theme: str | None = None
    role_id: str | None = None
    user_id: str | None = Field(default=None, description="被服务端忽略；身份取登录态")


class PrefsResponse(BaseModel):
    """账号偏好的读响应：`theme`（null = 跟随系统）+ 生效 `role_id` + 元数据。"""

    user_id: str
    theme: str | None = None
    role_id: str
    role_id_is_default: bool = True
    updated_at: float = 0.0


# ---------------- 入库 ----------------

class IngestRequest(BaseModel):
    sources: list[str] | None = None
    rebuild: bool = True


# ---------------- 账号体系请求模型 ----------------
#
# 只收 username + password：**没有** email / tel / 验证码字段（`AC-AU-3/7`），
# 「确认密码」是前端的事（两次输入一致性在本地就能判定，交给服务端只会多一次往返）。
# 长度约束放在端点里用业务校验（400 + 具体字段名），不用 pydantic 的 min_length
# —— 后者会变成 422 且文案是机器格式的，与规范要求的 400 + `message` 不一致。

class RegisterRequest(BaseModel):
    username: str = Field(..., description="用户名（3-32 位 [a-z0-9_-]，不区分大小写）")
    password: str = Field(..., description="密码（至少 10 位，不做 trim）")


class LoginRequest(BaseModel):
    username: str = Field(..., description="用户名（同注册，大小写不敏感）")
    password: str = Field(..., description="密码")


class ResetRequest(BaseModel):
    """找回密码的请求体（`AC-AU-56`：**只有这三个字段**）。

    刻意**没有** email / tel / code（验证码）字段：重置只靠「用户名 + 一次性恢复码」，
    不依赖任何外部信道（不收集手机号/邮箱、不发短信或邮件）。

    ⚠️ 三个字段**都给了 `""` 默认值**（而不是 `...` 必填）：缺字段必须由**业务层**判定，
    才能按 `AC-RC-3` 回 **400** ``{"error":"reset_fields_required"}``；用 pydantic 的必填
    会先被拦成 **422** 机器格式错误体 —— 同一个"缺字段"在规范里是 400，不是 422。
    """
    username: str = Field("", description="用户名（同注册，大小写不敏感）")
    recovery_code: str = Field("", description="注册时一次性给出的恢复码")
    new_password: str = Field("", description="新密码（10–128，规则同注册）")
