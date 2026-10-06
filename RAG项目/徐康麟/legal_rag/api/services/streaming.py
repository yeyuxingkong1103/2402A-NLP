# -*- coding: utf-8 -*-
"""SSE 流式响应的帧构造与包装 —— 从 ``api/app.py`` 拆出。

为什么单独一个文件
------------------
这两段是**纯逻辑**（拼帧、包流），只依赖 ``json`` 与 logger，不碰 FastAPI 的
路由装配。原先埋在 1,700 行的 ``app.py`` 里，改动 `stream_mode` 相关行为时
很难定位。拆出来后：
* `/chat` 的 sse 分支只需要 ``_sse_generator``；
* 帧格式（事件名 / 字段名）是**前端契约**，集中一处便于核对。

⚠️ 事件名（``delta`` / ``citations`` / ``done`` / ``error``）与字段名是前端契约，
改名要同步改 ``web/ui.html`` 与 ``tests/test_api.py`` 的 SSE 用例。
"""
from __future__ import annotations

import json
from typing import Any, AsyncIterator

from ...logging_setup import get_logger

__all__ = ["sse_frame", "sse_generator"]

logger = get_logger("legal_rag.api.services.streaming")


def sse_frame(event: str, data: Any) -> str:
    """拼一个 SSE 帧：``event: <名>`` + ``data: <JSON>`` + 空行分隔。"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def sse_generator(answer: Any, pieces: AsyncIterator[str]) -> AsyncIterator[str]:
    """把文本流包成 SSE：``delta`` 文本帧 + 结束时的 ``citations``/``done`` 帧。

    为什么要这层：裸文本流**拿不到引用**（引用在流开始前就算好了），
    而网页聊天两者都要。``stream_mode=text`` 保持与既往完全一致，只有 ``sse`` 走这里。
    """
    try:
        async for piece in pieces:
            if piece:
                yield sse_frame("delta", {"text": piece})
    except Exception as exc:  # noqa: BLE001 - 流中途失败也要让前端知道，而不是静默断流
        logger.exception("SSE 流式生成失败")
        yield sse_frame("error", {"error": f"{type(exc).__name__}: {exc}"})
        return
    yield sse_frame("citations", {"citations": list(answer.citations or []),
                                  "provider": answer.provider,
                                  "role_id": answer.role_id,
                                  "degraded": bool(getattr(answer, "degraded", False)),
                                  "degraded_reason": getattr(answer, "degraded_reason", "")})
    yield sse_frame("done", {"answer": answer.text})
