"""运维可视化接口：Redis 与 Milvus 监控数据（供前端可视化页使用）。"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.config import settings
from app.core.registry import get_embedder
from app.db.milvus_store import milvus_store
from app.db.redis_store import redis_store
from app.logging_conf import log
from app.schemas import MilvusSearchReq

router = APIRouter(prefix="/ops", tags=["ops"])


# ===== Redis 可视化 =====
@router.get("/redis/stats")
def redis_stats() -> dict:
    return redis_store.stats()


@router.get("/redis/keys")
def redis_keys(pattern: str = "*", limit: int = 200) -> list[dict]:
    return [redis_store.key_info(k) for k in redis_store.scan_keys(pattern, limit)]


@router.get("/redis/key/{key:path}")
def redis_key(key: str) -> dict:
    return redis_store.key_info(key)


@router.get("/redis/session/{user_id}/{role_id}")
def redis_session(user_id: str, role_id: int) -> dict:
    return {
        "messages": redis_store.get_messages(user_id, role_id),
        "meta": redis_store.get_session_meta(user_id, role_id),
    }


# ===== Milvus 可视化 =====
@router.get("/milvus/collections")
def milvus_collections() -> dict:
    return {"current": settings.milvus_collection, "collections": [settings.milvus_collection]}


@router.get("/milvus/stats")
def milvus_stats() -> dict:
    try:
        return milvus_store.stats()
    except Exception as exc:  # noqa: BLE001
        log.error("Milvus stats 失败: %s", exc)
        raise HTTPException(500, str(exc)) from exc


@router.get("/milvus/entities")
def milvus_entities(domain: str | None = None, limit: int = 50, offset: int = 0) -> list[dict]:
    try:
        return milvus_store.page_entities(domain, limit, offset)
    except Exception as exc:  # noqa: BLE001
        log.error("Milvus entities 失败: %s", exc)
        raise HTTPException(500, str(exc)) from exc


@router.post("/milvus/search")
def milvus_search(req: MilvusSearchReq) -> list[dict]:
    try:
        vector = get_embedder().encode_one(req.query)
        return milvus_store.search_hybrid(req.query, vector, req.top_k, req.domain)
    except Exception as exc:  # noqa: BLE001
        log.error("Milvus 检索预览失败: %s", exc)
        raise HTTPException(500, str(exc)) from exc
