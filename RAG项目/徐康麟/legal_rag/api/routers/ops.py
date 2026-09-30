# -*- coding: utf-8 -*-
"""运维端点：``/health`` ``/livez`` ``/metrics`` ``/roles`` —— 从 ``api/app.py`` 拆出。

这四条是**只读、无业务写入**的端点，闭包依赖只有 ``app_state``，因此是验证
"路由搬出 ``create_app``"这套模式的最佳起点。

⚠️ 三条端点的**分工是刻意设计的**，别把它们合并：
* ``/livez``  → 存活（liveness）：进程活着就 200，**不碰任何依赖**，O(1) 且不占线程池；
* ``/health`` → 就绪与体检（readiness/diagnostics）：依赖、降级、并发、GPU、条数；
  **默认读快照**（O(1)），``?deep=1`` 才现采 —— 详见下方注释；
* ``/metrics``→ 只渲染指标注册表，**不做任何**检索/生成/后端探测。
"""
from __future__ import annotations

import time

from fastapi import APIRouter, Depends
from fastapi.responses import PlainTextResponse

from ... import metrics as M
from ..concurrency import run_blocking
from ..deps import AppState
from ...roles import ROLE_LIBRARY
from ..schemas import RoleResponse
from ..dependencies import get_app_state

router = APIRouter()


# ---------- 健康 ----------
#
# D4 方案 A（2026-09-28 拍板）：**默认读快照**，`?deep=1` 才现采。
#   旧的"每次都现采"在依赖不可达时单次 = 探针超时（实测 Redis 挂掉 2.2 s），
#   还占一个工作线程 ⇒ 8 线程下吞吐被钉在 ~3.6 QPS。
#   快照由 `lifespan` 里的周期采样器每轮刷新；采样器停了就**回落现采**并标 `stale=true`。
#   响应里 `check` / `snapshot_age_seconds` / `stale` 三个键说明这次是怎么来的。
@router.get("/health")
async def health(deep: int = 0,
                 app_state: AppState = Depends(get_app_state)) -> dict:
    if deep or app_state.engine is None:
        # 显式现采，或引擎还没装配（lazy 装配：这时没有后台快照可言）
        return await run_blocking(app_state.health_payload,
                                  limiter=app_state.blocking_limiter(),
                                  deep=bool(deep))
    # 快照路径：**O(1)、不进线程池**（这正是 D4 要的效果）
    return app_state.health_payload()


# ---------- 存活探针（**与 /health 分工**）----------
#
# ⚠️ 真机实测（2026-09-26，见 docs/LOAD-TEST.md §3）：`/health` 是**深度体检** ——
#    它每次都现采一次后端探针（redis/milvus/ollama + GPU），
#    而探针成本在依赖**不可达**时等于**超时值**（实测 Redis 挂掉时单次 2.2 s），
#    且它占用一个工作线程（8 线程 ⇒ 吞吐被钉在 ~3.6 QPS，并发 32 时 P50 涨到 9.3 s）。
#    ⇒ **LB / K8s / 监控的高频探活绝不能用 `/health`**：依赖一挂，探活会把线程池吃光，
#      连 `/chat` 一起拖垮（"雪崩放大"）。
#
#    `/livez` 只回答一个问题："这个进程还活着、还能收请求吗" ——
#    **不碰任何依赖**，因此永远是 O(1) 且不占线程池（不是阻塞调用，直接返回）。
#    语义分工：
#      * `/livez`  → 存活（liveness）：进程活着就 200；**别在这里判断依赖**
#      * `/health` → 就绪与体检（readiness/diagnostics）：依赖、降级、并发、GPU、条数
@router.get("/livez")
async def livez(app_state: AppState = Depends(get_app_state)) -> dict:
    return {
        "status": "ok",
        "uptime_seconds": round(time.time() - app_state.started_at, 3),
        # 明确标注"不探测依赖"，避免调用方把它当成就绪判据
        "checks": "skipped",
        "note": "存活探针：不探测任何依赖；就绪与依赖状态请用 /health",
    }


# ---------- 指标 ----------
@router.get("/metrics")
async def metrics() -> PlainTextResponse:
    """Prometheus 文本。

    **必须轻**：只渲染注册表 —— 系统层样本由 ``lifespan`` 里的周期采样器持续发布，
    业务样本由各链路自己采集；这里**不做**任何检索/生成/后端探测。
    """
    body = M.render_prometheus()
    return PlainTextResponse(content=body, media_type="text/plain; version=0.0.4")


# ---------- 角色 ----------
@router.get("/roles", response_model=list[RoleResponse])
async def roles() -> list[RoleResponse]:
    return [RoleResponse(**role.__dict__) for role in ROLE_LIBRARY.values()]
