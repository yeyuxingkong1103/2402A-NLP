# -*- coding: utf-8 -*-
"""server —— RAG 服务入口（同名包）。

在链路中的位置（最外层）：
    浏览器(static/index.html) → 【本包】 → pipeline（离线构建）
                                        → retrieval（在线检索）
                                        → roleplay（角色扮演）
                                        → Milvus / Ollama

本包只做三件事，不掺业务算法：
    1. 把 HTTP 请求翻译成对 pipeline / retrieval / roleplay 的调用
    2. 管住"同一时间只有一个构建任务"这个全局状态
    3. 把静态页面挂到根路径，让前后端同源、免跨域

为什么改成了包：
    原 server.py 有 672 行（其中注释 215 行），超出"单文件 300 行"的上限。
    按业务分组拆成 config / state / registry / security / answer +
    四个 routes_* 模块后，每个文件都在 300 行以内。

关键设计：接口全部挂在各 routes_* 模块的 APIRouter 上，
    由本文件统一 include 到 app —— app 的对外路由表与拆分前**逐条一致**
    （启动命令 `uvicorn backend.server:app` 与 `uvicorn server:app` 都照旧可用）。

包内分工：
    config.py           路径、模型地址、拒答文案、日志配置
    state.py            构建状态
    registry.py         文档登记表读写
    security.py         文件名校验
    answer.py           生成带引用的答案
    routes_build.py     上传与构建进度
    routes_search.py    检索与问答
    routes_kb.py        知识库管理
    routes_roleplay.py  角色扮演
    本文件              组装 app、挂载路由与静态目录
"""
from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

try:
    from ..pipeline import COLLECTION
    from ..vector_store import milvus_health
except ImportError:
    from pipeline import COLLECTION
    from vector_store import milvus_health

from .config import LLM_MODEL, STATIC_DIR
from .registry import load_docs
from .routes_build import router as build_router
from .routes_kb import router as kb_router
from .routes_roleplay import router as roleplay_router
from .routes_search import router as search_router

app = FastAPI(title="RAG 知识库问答系统")

app.include_router(build_router)
app.include_router(search_router)
app.include_router(kb_router)
app.include_router(roleplay_router)


# /api/health 不属于任何业务分组，直接挂在 app 上
@app.get("/api/health")
def health():
    """健康检查：Milvus 连得上就是 ok，连不上返回 degraded（而不是 500）。

    为什么用 degraded 而不是直接报错：
        Milvus 挂了时检索/上传确实不可用，但服务进程本身还活着、
        静态页面也还能打开。返回 degraded 能让调用方知道
        "是依赖服务的问题，不是应用崩了"，排障方向完全不同。
    """
    connected, detail = milvus_health()
    return {
        "status": "ok" if connected else "degraded",
        "collection": COLLECTION,
        "documents": len(load_docs()),
        "llm_model": LLM_MODEL,
        "vector_backend": "milvus",
        "milvus_uri": os.getenv("MILVUS_URI", "http://127.0.0.1:19530"),
        "milvus_connected": connected,
        "milvus_detail": detail,
    }



# 必须放在所有 API 路由之后：挂到 "/" 会吃掉根路径下的所有匹配，
# 若写在前面，后面定义的具体路由就再也匹配不到了。
# html=True 让 / 直接返回 index.html。
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
