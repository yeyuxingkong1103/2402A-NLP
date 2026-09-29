# -*- coding: utf-8 -*-
"""健康检查：五项依赖逐项探活。"""
import re
import time
from pathlib import Path

import httpx
from fastapi import APIRouter
from sqlalchemy import text

from ..core import config
from ..core.db import engine
from ..core.logging import get_logger
from ..services import chain
from ..services.qdrant_store import get_client

router = APIRouter(tags=["健康检查"])
log = get_logger("health")


def _check_mysql() -> dict:
    t = time.time()
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"ok": True, "ms": round((time.time() - t) * 1000)}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {str(e)[:150]}"}


def _check_redis() -> dict:
    t = time.time()
    try:
        import redis
        client = redis.Redis(
            host=config.REDIS_HOST, port=config.REDIS_PORT, db=config.REDIS_DB,
            decode_responses=True,
            encoding_errors="replace",   # Windows 版 Redis 的 INFO 含 GBK 字节，必须容错
            socket_connect_timeout=3,
        )
        client.ping()
        version = client.info("server").get("redis_version")
        client.close()
        return {"ok": True, "ms": round((time.time() - t) * 1000), "version": version}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {str(e)[:150]}"}


def _check_qdrant() -> dict:
    """Qdrant 探活。

    ⚠️ `VECTOR_STORE=milvus` 时**必须跳过**（2026-09-16 修）：
    `get_client()` 会打开嵌入式 Qdrant 并**持有独占文件锁**，若在此无条件调用，
    即便向量库已切到 Milvus，后端仍然占着 Qdrant 的锁 ——
    而「换 Milvus 以解除单 worker 限制」正是这次接入的核心收益之一。
    健康检查又恰恰是 nginx/前端会轮询的端点，等于每次轮询都把锁重新拿住。
    """
    if config.VECTOR_STORE == "milvus":
        return {"ok": True, "skipped": True,
                "reason": "VECTOR_STORE=milvus，未启用 Qdrant（不占用其独占锁）"}
    t = time.time()
    try:
        names = [c.name for c in get_client().get_collections().collections]
        return {"ok": True, "ms": round((time.time() - t) * 1000), "collections": names}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {str(e)[:150]}"}


def _check_neo4j() -> dict:
    """Neo4j 探活（多路召回的图谱路用）。

    与 Milvus 同样处理：**未启用就标记 skipped**，不让可选组件拖红整体健康检查。
    `MULTI_RECALL_ENABLED=0` 时图谱路本就不参与，无需探活。
    """
    if not config.MULTI_RECALL_ENABLED:
        return {"ok": True, "skipped": True, "reason": "多路召回未启用"}
    from ..services import graph_store
    res = graph_store.health()
    res["active"] = True
    return res


def _check_milvus() -> dict:
    """Milvus 探活。

    未启用（`VECTOR_STORE=qdrant`）时标记为 `skipped` 而非失败 ——
    Milvus 是**可选**后端（需 Docker），主链路不依赖它。

    但**启用时**（`milvus`/`both`）探活失败会让整体 `ok` 变红，
    这是有意的：此时 Milvus 就是主向量库，不可达必须能被立刻发现。
    """
    if config.VECTOR_STORE == "qdrant":
        return {"ok": True, "skipped": True,
                "reason": "VECTOR_STORE=qdrant，未启用 Milvus"}
    from ..services import milvus_store
    res = milvus_store.health()
    res["active"] = True
    return res


def _check_ollama() -> dict:
    t = time.time()
    base = config.OLLAMA_BASE_URL.rstrip("/")
    if not re.match(r"^https?://", base):
        return {"ok": False, "error": f"OLLAMA_BASE_URL 仅支持 http(s)，当前: {base[:60]}"}
    try:
        r = httpx.get(f"{base}/models", timeout=5, follow_redirects=False)
        r.raise_for_status()
        models = [m["id"] for m in r.json().get("data", [])]
        return {
            "ok": True, "ms": round((time.time() - t) * 1000),
            "models": models, "llm_ready": config.LLM_MODEL in models,
        }
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {str(e)[:150]}"}


def _check_models() -> dict:
    embed = Path(config.EMBED_MODEL_PATH)
    rerank = Path(config.RERANK_MODEL_PATH)
    return {
        "ok": embed.exists() and rerank.exists(),
        "embed": {"path": str(embed), "exists": embed.exists(),
                  "sparse_linear": (embed / "sparse_linear.pt").exists()},
        "rerank": {"path": str(rerank), "exists": rerank.exists()},
    }


def _check_gpu() -> dict:
    try:
        import torch
        if not torch.cuda.is_available():
            return {"ok": False, "error": "CUDA 不可用"}
        free, total = torch.cuda.mem_get_info()
        return {
            "ok": True,
            "device": torch.cuda.get_device_name(0),
            "vram_total_gb": round(total / 1024 ** 3, 2),
            "vram_free_gb": round(free / 1024 ** 3, 2),
        }
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {str(e)[:150]}"}


@router.get("/health", summary="依赖探活")
def health(deep: bool = False):
    checks = {
        "mysql": _check_mysql(),
        "redis": _check_redis(),
        "qdrant": _check_qdrant(),
        "milvus": _check_milvus(),
        "neo4j": _check_neo4j(),
        "ollama": _check_ollama(),
    }
    if deep:
        # torch 导入较慢（数秒），仅在 ?deep=1 时检查
        checks["models"] = _check_models()
        checks["gpu"] = _check_gpu()

    return {
        "ok": all(v.get("ok") for v in checks.values()),
        # 便于排查「到底在跑哪条链路 / 哪个向量库」
        "active": {
            "chain": chain.active_backend(),
            "vector_store": config.VECTOR_STORE,
            "llm_model": config.LLM_MODEL,
            "multi_recall": config.MULTI_RECALL_ENABLED,
        },
        "checks": checks,
    }
