# -*- coding: utf-8 -*-
"""FastAPI 入口：CORS + 路由挂载 + 启动建表/种角色。

启动：
    D:\\anaconda3\\envs\\rag_env\\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000
⚠️ 不要加 --workers（Qdrant 嵌入式模式持有独占文件锁）。
"""
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .core.config import BASE_DIR, DB_URL, MYSQL_DB, WARMUP_ENABLED
from .core.db import create_all
from .core.logging import get_logger, setup_logging
from .routers import auth, characters, chat, conversations, health, kb, search
from .seed import seed_characters
from .services.qdrant_store import close_client

setup_logging()
log = get_logger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("=" * 60)
    log.info("RAGLoRA 后端启动中 | 项目目录 %s", BASE_DIR)
    try:
        create_all()
        log.info("建表完成 | 数据库 %s", MYSQL_DB)
    except Exception as e:
        log.error("建表失败: %s", e)
        raise
    created = seed_characters()
    log.info("内置角色种入 %d 个", created)

    # Redis Set 分类索引全量重建（幂等）：自愈「写路径漏更索引」的任何残留
    try:
        from .services import redis_extra
        indexed = redis_extra.rebuild_category_index()
        log.info("角色分类索引重建 %d 项", indexed)
    except Exception as e:
        log.warning("分类索引重建失败（索引查询会回源 MySQL，不影响启动）: %s", str(e)[:150])
    log.info("启动完成 -> http://127.0.0.1:8000/docs")
    log.info("=" * 60)

    # 后台预热模型：首次请求不必等 7 秒的加载时间
    # 内存吃紧的机器可用 RAGLORA_WARMUP=0 关闭，改为按需加载
    def _warmup():
        import time
        t0 = time.time()
        try:
            from .services import embed, rerank
            embed.warmup("cpu")
            rerank.warmup()
            log.info("模型预热完成（bge-m3 + 精排）耗时 %.1fs", time.time() - t0)
        except Exception as e:
            log.warning("模型预热失败（不影响使用，首请求会慢些）: %s", str(e)[:150])

    if WARMUP_ENABLED:
        threading.Thread(target=_warmup, daemon=True, name="warmup").start()
    else:
        log.info("已关闭模型预热（RAGLORA_WARMUP=0），模型将在首次使用时加载")
    yield
    close_client()
    try:
        from .services import embed, rerank
        embed.release_gpu()
        rerank.release()
    except Exception:
        pass
    log.info("RAGLoRA 后端已停止")


app = FastAPI(
    title="RAGLoRA · 多用户多角色 RAG 系统",
    description="基于 RAG 的角色扮演系统：医生 / 律师等角色，各自独立知识库与人格模板。",
    version="1.0.0",
    lifespan=lifespan,
)

# 用 Bearer Token 鉴权（非 Cookie），故不开 allow_credentials
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/api")
app.include_router(auth.router, prefix="/api")
app.include_router(characters.router, prefix="/api")
app.include_router(kb.router, prefix="/api")
app.include_router(search.router, prefix="/api")
app.include_router(conversations.router, prefix="/api")
app.include_router(chat.router, prefix="/api")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)
