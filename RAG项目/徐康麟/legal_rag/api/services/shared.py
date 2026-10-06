# -*- coding: utf-8 -*-
"""跨路由复用的 HTTP 小工具 —— 从 ``api/app.py`` 拆出。

这里放"多个路由模块都要用、但不属于任何单一业务"的助手。判断标准：
* 它只依赖 ``Request`` 或纯数据，**不依赖 ``AppState``** —— 依赖 AppState 的
  助手属于具体业务，应放到对应的 services 模块里（如 ``services/uploads.py``）。
"""
from __future__ import annotations

import json
from typing import Any

from fastapi import Request

from ...logging_setup import get_logger
from ..auth import extract_token

__all__ = ["path_label", "request_token", "load_citations"]

logger = get_logger("legal_rag.api.services.shared")


def path_label(request: Request) -> str:
    """指标标签用**路径模板**（避免每个 session_id/doc_id 都变成一条时间序列）。

    取 ``request.scope["route"].path``（FastAPI 注册的模板，形如
    ``/sessions/{session_id}``），拿不到才回落到实际 URL —— 这是本项目的既有约定，
    见 ``docs/ARCHITECTURE.md`` §7「路径模板」。**不要**改成 ``request.url.path``，
    那会让每个会话 ID 变成一条时间序列。
    """
    route = request.scope.get("route")
    return getattr(route, "path", None) or request.url.path


def request_token(request: Request) -> str | None:
    """取请求里的会话令牌：**HttpOnly Cookie 优先**，其次 ``Authorization: Bearer``。

    Cookie 名默认 ``lr_session``；Bearer 只是给脚本/curl 用的同一枚令牌的另一种带法
    （值本身仍然是服务端签发的高熵串，安全性不因多一种带法而降低）。
    """
    auth_state = getattr(getattr(request.app, "state", None), "app_state", None)
    auth_service = getattr(auth_state, "auth", None)
    cookie_name = auth_service.cookie_name if auth_service is not None else "lr_session"
    return extract_token(request.cookies.get(cookie_name),
                         request.headers.get("authorization"))


def load_citations(raw: Any) -> list[dict]:
    """关系库里存的引用来源（JSON 文本）→ 列表；坏数据按"无引用来源"处理。

    为什么容忍坏数据：历史消息可能是更早版本写的，或手工改过库；
    **一条历史消息解析失败不该让整个会话列表 500**（用户只是翻历史）。
    """
    if isinstance(raw, list):
        return list(raw)
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        logger.warning("历史消息的 citations 解析失败，按「无引用来源」处理")
        return []
    return list(parsed) if isinstance(parsed, list) else []
