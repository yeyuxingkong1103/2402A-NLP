"""多轮对话的传输模型。

契约见 `specs/010-multiturn-chat/contracts/chat-api.md`。

---

## 为什么单独一个文件而不是并进 `schemas.py`

`backend/api/schemas.py` 的字段名被 SC-009 锁死（"MUST NOT 增删改"）——
它是 `docs/05` §3.1.3 的落点，`/ask` 的对外契约。把新特性的模型混进去，
会让"这个文件能不能动"这个问题失去答案。

`corpus_routes.py` 已经是"每特性一对文件"的先例，本模块沿用。

## ⚠️ 校验只在这一层

非空、长度上限全部在 Pydantic 层拦下，**MUST NOT 在服务层补判**。
补判会让边界值判定散落两处，而 `backend/api/validate.py` 的既有设计
明确要求判定**只此一处**（`schemas.py` 的模块注释记录了这条）。

Pydantic 校验失败发生在进入路由函数体**之前**，因此路由里 `try/except`
是捕获不到的 —— 它由 `backend/api/errors.py` 的异常处理器按路径分派
（见那里的 `_validation_error`）。
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from . import QUESTION_MAX_LEN

__all__ = [
    "CreateSessionRequest",
    "RenameSessionRequest",
    "SendMessageRequest",
    "ChatMessage",
    "HistoryResponse",
    "SessionResponse",
]


class CreateSessionRequest(BaseModel):
    """`POST /api/chat/session` 的请求体。

    ⚠️ **`user_id` 可选**（Q5 裁决）。不提供或为 `null` 时由服务端自动生成。
    使用者 MUST NOT 被强制先想一个标识才能开始对话 ——
    而当前系统没有登录体系，这个字段不参与任何校验。
    """

    user_id: str | None = None

    @field_validator("user_id")
    @classmethod
    def _strip(cls, value: str | None) -> str | None:
        """去首尾空白；**空串归一成 `None`**（而不是报错）。

        一个长度为 0 的显示名与"没提供"是同一件事。让它落库只会让
        界面显示一片空白，而使用者明明什么都没填。
        """

        if value is None:
            return None
        stripped = value.strip()
        return stripped or None


class RenameSessionRequest(BaseModel):
    """`PATCH /api/chat/session/{session_id}` 的请求体。

    与创建不同，这里 `user_id` **必填且非空** —— 改名的语义就是"改成什么"，
    一个空值不对应任何意图。放行它会写进一个空白标识，
    与"自动生成"的语义混在一起（contracts/chat-api.md §2）。
    """

    user_id: str

    @field_validator("user_id")
    @classmethod
    def _non_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("user_id 不能为空白")
        return stripped


class SendMessageRequest(BaseModel):
    """`POST /api/chat/message` 的请求体。"""

    session_id: str
    content: str

    @field_validator("session_id")
    @classmethod
    def _session_non_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("session_id 不能为空")
        return stripped

    @field_validator("content")
    @classmethod
    def _content_bounds(cls, value: str) -> str:
        """去空白后非空，且长度不超过 `QUESTION_MAX_LEN`。

        ⚠️ 上限**复用** `/ask` 的 `QUESTION_MAX_LEN` 而不另起一个数值相同的常量。

        理由：它是**产品边界的显式表达**（`docs/05` §3.1.2：M1 定义的是单轮、
        非专业的口语化提问），而"多轮对话里的一条消息"与"一次提问"
        在产品上是同一件事。两份定义会漂移 —— 而漂移的表现是
        "同一个问题，在 `/ask` 能问、在 chat 被拒"，无人能解释为什么。

        ⚠️ 超长 MUST 在写入之前被拒绝。先落库再报错会在库里留下一条
        永远不会进入上下文的记录（spec 的 Edge Case）。
        """

        stripped = value.strip()
        if not stripped:
            raise ValueError("消息内容不能为空")
        if len(stripped) > QUESTION_MAX_LEN:
            raise ValueError(f"消息过长，请控制在 {QUESTION_MAX_LEN} 个字符以内")
        return stripped


class ChatMessage(BaseModel):
    """历史里的一条消息。字段与存储结构逐项对应。"""

    role: str
    content: str
    timestamp: int
    message_id: str


class HistoryResponse(BaseModel):
    """`GET /api/chat/history/{session_id}` 的响应。

    `total` 是**会话当前的消息总条数**，与请求的 `limit` 无关 ——
    前端据此判断"还有更多"。
    """

    session_id: str
    total: int
    messages: list[ChatMessage]


class SessionResponse(BaseModel):
    """创建与改名的共用响应。

    ⚠️ **`user_id` MUST 回传。** 创建时它可能是自动生成的 ——
    不回传的话，使用者在那种情形下**无从得知**自己的标识是什么，
    更谈不上修改它（contracts/chat-api.md §1）。
    """

    session_id: str
    user_id: str
    created_at: int
    last_active_at: int
    expires_in: int = Field(description="会话存活时长（秒），从本次操作起算")
