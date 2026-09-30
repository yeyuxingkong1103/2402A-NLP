"""数据对象 used by the shared chat orchestration layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.api.v1.chat_schemas import ChatRequest


@dataclass(slots=True, frozen=True)
class PreparedChat:
    """一次请求在生成答案前的统一中间结果。

    Streaming 和非 streaming 必须共享这份结果，否则两条路径的检索、
    重排和拒答判断可能出现分叉。
    """

    request: ChatRequest
    conversation_id: str
    rewritten_query: str
    history: list[dict[str, Any]]
    retrieved_chunks: Any
    final_chunks: Any

    @property
    def refused(self) -> bool:
        """没有最终候选时，服务层进入安全拒答分支。"""
        return not self.final_chunks
