"""健康检查接口。"""
from __future__ import annotations

from fastapi import APIRouter

from app.config import settings
from app.db.milvus_store import milvus_store
from app.db.mongo_store import mongo_store
from app.db.redis_store import redis_store

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "env": settings.app_env,
        "llm_provider": settings.llm_provider,
        "rag_engine": settings.rag_engine,
        "components": {
            "redis": redis_store.available,
            "mongo": mongo_store.enabled,
            "milvus": _milvus_ok(),
        },
    }


def _milvus_ok() -> bool:
    try:
        milvus_store.count()
        return True
    except Exception:  # noqa: BLE001
        return False
