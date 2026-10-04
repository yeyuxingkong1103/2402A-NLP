"""FastAPI 应用入口。

启动方式：``uvicorn app.main:app --host 0.0.0.0 --port 8000``
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.api import chat, health, kb, memory, ops, roles
from app.config import settings
from app.db.milvus_store import milvus_store
from app.db.mysql_store import init_mysql
from app.logging_conf import log


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("启动中... env=%s provider=%s engine=%s",
             settings.app_env, settings.llm_provider, settings.rag_engine)
    try:
        init_mysql()
    except Exception as exc:  # noqa: BLE001
        log.error("MySQL 初始化失败: %s", exc)
    try:
        col = milvus_store.collection()
        if settings.auto_ingest and col.num_entities == 0 and os.path.isdir(settings.laws_dir):
            from app.ingest.pipeline import ingest_directory

            ingest_directory(settings.laws_dir, domain="law")
    except Exception as exc:  # noqa: BLE001
        log.error("Milvus 初始化失败: %s", exc)
    log.info("启动完成")
    yield
    log.info("服务关闭")


app = FastAPI(title="RAG 多角色扮演系统", version=__version__, lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

for module in (chat, roles, kb, memory, health, ops):
    app.include_router(module.router)


@app.get("/")
def root() -> dict:
    return {"app": "RAG 多角色扮演系统", "version": __version__, "docs": "/docs"}
