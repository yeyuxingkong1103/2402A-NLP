"""记忆接口：清空会话、查看短期记忆与会话列表。"""
from __future__ import annotations

from fastapi import APIRouter

from app.db.redis_store import redis_store

router = APIRouter(tags=["memory"])


@router.post("/clear")
def clear(user_id: str, role_id: int = 1) -> dict:
    redis_store.clear_session(user_id, role_id)
    return {"ok": True}


@router.get("/memory/{user_id}/{role_id}")
def get_memory(user_id: str, role_id: int) -> dict:
    return {
        "messages": redis_store.get_messages(user_id, role_id),
        "meta": redis_store.get_session_meta(user_id, role_id),
    }


@router.get("/sessions/{user_id}")
def list_sessions(user_id: str) -> list[str]:
    return redis_store.list_sessions(user_id)
