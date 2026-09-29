# -*- coding: utf-8 -*-
"""server/routes_roleplay.py —— 角色扮演接口。

在链路中的位置：
    浏览器 → 【本文件】 → backend/roleplay 包（角色 / 会话 / 记忆 / 检索 / 生成）

七个接口：
    POST /api/roleplay/chat                            执行一次角色聊天
    GET  /api/roleplay/roles                           角色列表
    POST /api/roleplay/roles                           保存自定义角色
    POST /api/roleplay/sessions                        新建会话
    GET  /api/roleplay/sessions                        会话列表
    GET  /api/roleplay/sessions/{session_id}/messages  会话历史

本文件只做"HTTP 协议转换 + 参数校验 + 异常映射"，不含任何角色扮演逻辑。
异常映射：ValueError -> 400（参数问题），其他 -> 500（服务端故障），
这样前端能区分"我传错了"和"服务器炸了"。
"""
from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

try:
    from .. import roleplay
except ImportError:
    import roleplay

from .config import logger

router = APIRouter()

class RoleplayChatRequest(BaseModel):
    """角色扮演对话请求体。"""

    user_id: str = "anonymous"       # 缺省匿名：前端演示时不需要先登录
    user_name: str | None = None
    role_id: str = "friend"
    session_id: str | None = None    # 不传则自动新建会话
    message: str

@router.post("/api/roleplay/chat")
def roleplay_chat(request: RoleplayChatRequest):
    """执行一次角色扮演对话（全部逻辑在 roleplay.py）。

    异常映射：
        ValueError -> 400（参数问题，如角色不存在、消息为空）
        其他异常   -> 500（服务端故障）
        这样前端能区分"我传错了"和"服务器炸了"。
    """
    try:
        return roleplay.chat(request.user_id, request.role_id, request.message, request.session_id, request.user_name)
    except ValueError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
    except Exception as exc:
        logger.exception("roleplay chat failed")
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)

@router.get("/api/roleplay/roles")
def roleplay_roles():
    """列出内置角色和用户自定义角色。"""
    return {"roles": roleplay.list_roles()}

class RoleRequest(BaseModel):
    """自定义角色请求体，字段对应角色卡的四要素。"""

    id: str
    name: str
    description: str = ""
    persona: str                                        # 角色是谁（必填）
    style: str = ""                                     # 怎么说话
    safety_notice: str = ""                             # 安全边界提示
    knowledge_sources: list[str] = Field(default_factory=list)  # 限定只引用哪些文档；空=全库

    # 用 Field(default_factory=list) 而非 = []：可变默认值在 Python 里是共享对象，
    # 多个请求会互相污染，这是必须避开的经典陷阱


@router.post("/api/roleplay/roles")
def roleplay_save_role(request: RoleRequest):
    """保存（新建或覆盖）一个自定义角色。"""
    try:
        return {"ok": True, "role": roleplay.save_role(request.model_dump())}
    except ValueError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)

class SessionRequest(BaseModel):
    """新建会话请求体。"""

    user_id: str = "anonymous"
    role_id: str = "friend"
    title: str = ""

@router.post("/api/roleplay/sessions")
def roleplay_create_session(request: SessionRequest):
    """为某用户在某角色下新建一个会话。"""
    try:
        return roleplay.create_session(request.user_id, request.role_id, request.title)
    except ValueError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)

@router.get("/api/roleplay/sessions")
def roleplay_sessions(user_id: str = "anonymous", limit: int = 50):
    """列出某用户的会话（默认最多 50 条）。"""
    return {"sessions": roleplay.list_sessions(user_id, limit)}

@router.get("/api/roleplay/sessions/{session_id}/messages")
def roleplay_messages(session_id: str, user_id: str = "anonymous", limit: int = 100):
    """查看某会话的历史消息。

    越权校验：
        查出会话后比对 user_id 是否一致，不一致返回 403。
        会话 id 是可猜测的短字符串，不做这个校验的话，任何人改一下 id
        就能读到别人的聊天记录 —— 这是隐私问题，不是可选项。
    """
    session = roleplay.get_session(session_id)
    if not session:
        return JSONResponse({"ok": False, "error": "会话不存在"}, status_code=404)
    if session["user_id"] != user_id:
        return JSONResponse({"ok": False, "error": "无权访问该会话"}, status_code=403)
    return {"messages": roleplay.history(session_id, limit)}
