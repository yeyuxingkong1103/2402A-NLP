# -*- coding: utf-8 -*-
"""
收藏接口（增量功能）

设计要点：
    1. 收藏绑定当前登录用户（user_id），持久化 MySQL（user_favorites 表），
       清除浏览器缓存、换浏览器都不丢失。
    2. 收藏存的是「问题 + 答案 + 溯源」整条快照，而不是只存一个 ID ——
       知识库重建后，收藏的内容仍能完整回看，不会变成空壳。
    3. 快照以服务端 chat_messages 为准（前端只传 message_id），
       避免前端传来残缺或被改动过的内容。
    4. 同一用户对同一条问答重复收藏是幂等的，不报错、不产生重复行。

接口：
    POST   /api/favorite                收藏一条问答
    DELETE /api/favorite/{message_id}   取消收藏
    GET    /api/favorite/list           我的收藏列表
    GET    /api/favorite/ids            我收藏过的 message_id 列表

边界：只读写 user_favorites / chat_messages，不触碰知识库与 RAG 链路。
"""

from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.db import mysql
from backend.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/favorite", tags=["收藏"])

FAVORITE_LIMIT_MAX = 500


class FavoriteCreate(BaseModel):
    """收藏请求：只传 message_id，内容由服务端从历史记录里取快照"""

    user_id: str = Field(..., min_length=1, description="当前登录用户 ID")
    message_id: int = Field(..., ge=1, description="要收藏的问答消息 ID（来自提问响应）")
    session_id: str = Field("", description="来源会话 ID，可省略")
    role: str = Field("student", description="收藏时使用的身份，仅作记录")


def _require_user(user_id: str) -> str:
    if not user_id or not user_id.strip():
        raise HTTPException(status_code=400, detail="缺少 user_id，请先登录")
    return user_id.strip()


@router.post("", summary="收藏一条问答")
def add_favorite(request: FavoriteCreate) -> Dict[str, Any]:
    """
    收藏一条问答（问题 + 答案 + 溯源整条快照）。

    重复收藏同一条不报错，返回 already=true，前端据此把按钮置为已收藏态。
    """
    user_id = _require_user(request.user_id)

    if not mysql.get_user_by_id(user_id):
        raise HTTPException(status_code=404, detail="用户不存在，请重新登录")

    message = mysql.get_chat_message(message_id=request.message_id, user_id=user_id)
    if not message:
        raise HTTPException(
            status_code=404,
            detail="该问答不在你的历史记录中，无法收藏（请刷新后重试）",
        )

    result = mysql.add_favorite(
        user_id=user_id,
        message_id=request.message_id,
        session_id=message.get("session_id") or request.session_id,
        question=message.get("question") or "",
        answer=message.get("answer") or "",
        sources=message.get("sources") or [],
        role=message.get("role") or request.role or "student",
    )
    logger.info(
        "收藏问答 | 用户=%s | 消息ID=%s | 重复=%s",
        user_id, request.message_id, result.get("already"),
    )
    return {
        "favorite_id": result["id"],
        "message_id": request.message_id,
        "already": bool(result.get("already")),
        "message": "已在收藏中" if result.get("already") else "已加入我的收藏",
    }


@router.delete("/{message_id}", summary="取消收藏")
def remove_favorite(message_id: int, user_id: str) -> Dict[str, Any]:
    """取消收藏（按用户 + 消息 ID 精确取消，只能取消自己的）"""
    user_id = _require_user(user_id)
    removed = mysql.remove_favorite(user_id=user_id, message_id=message_id)

    logger.info("取消收藏 | 用户=%s | 消息ID=%s | 是否删除=%s", user_id, message_id, bool(removed))
    return {
        "message_id": message_id,
        "removed": bool(removed),
        "message": "已取消收藏" if removed else "该问答本就不在收藏中",
    }


@router.get("/list", summary="我的收藏列表")
def list_favorites(user_id: str, limit: int = 200) -> Dict[str, Any]:
    """列出当前用户的全部收藏，按收藏时间倒序"""
    user_id = _require_user(user_id)
    limit = max(1, min(int(limit or 200), FAVORITE_LIMIT_MAX))

    items: List[Dict[str, Any]] = mysql.list_favorites(user_id, limit=limit)
    return {"user_id": user_id, "total": len(items), "items": items}


@router.get("/ids", summary="我收藏过的消息 ID 列表")
def favorite_ids(user_id: str) -> Dict[str, Any]:
    """
    返回当前用户已收藏的全部 message_id。

    前端渲染历史记录时用它一次性判断「哪些答案已收藏」，
    避免为每条答案单独请求一次收藏状态。
    """
    user_id = _require_user(user_id)
    items = mysql.list_favorites(user_id, limit=FAVORITE_LIMIT_MAX)
    ids = [int(item["message_id"]) for item in items if item.get("message_id")]
    return {"user_id": user_id, "message_ids": ids}