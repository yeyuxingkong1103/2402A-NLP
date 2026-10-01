"""健康检查：探测 LLM / Embedding / Milvus / SQL / 短期记忆 连通状态。"""
from __future__ import annotations

from fastapi import APIRouter, Request

from ..schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse, response_model_exclude_defaults=True)
def health(request: Request):
    p = request.app.state.pipeline
    components = {
        "llm": "ok" if p.llm.ping() else "unavailable",
        "embedding": "ok" if p.embedding.ping() else "unavailable",
        "milvus": "ok" if p.milvus.ping() else "unavailable",
        "sql": "ok" if p.sql.ping() else "unavailable",
        # 不能写死 "ok"：Redis 挂着时这里必须报出来，否则前端绿点 + 每轮对话 500。
        # describe() 会返回 degraded（已降级到进程内内存，能用但不持久）。
        "memory": p.memory.describe(),
    }
    all_ok = all(v == "ok" for v in components.values())
    result: dict = {"status": "ok" if all_ok else "degraded", "components": components}
    # 只说 degraded 不够用：降级的代价是实打实的（重启后上下文全丢），得能看见「为什么」。
    # 只有降级实现带 status_detail（进程内实现没有话可说）。
    detail = getattr(p.memory, "status_detail", None)
    if detail and (reason := detail()):
        result["details"] = {"memory": reason}
    return result
