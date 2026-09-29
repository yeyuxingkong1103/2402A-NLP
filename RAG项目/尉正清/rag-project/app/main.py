# app/main.py
"""多角色 RAG 角色扮演系统 —— FastAPI 入口。

启动时只做轻量初始化（建库建表 + 写入角色种子数据）；
BGE-M3、Reranker 等重模型在首次请求时懒加载，避免启动阻塞。
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api import ROUTERS
from app.config import settings
from app.config.logging_conf import setup_logging
import logging

logger = logging.getLogger(__name__)

# 在创建 app 之前配置好日志，保证启动阶段的输出也能落盘
setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 60)
    logger.info("启动 %s v%s", settings.LLM_MODEL, "3.0")
    try:
        from app.db import mysql_conn
        from app.core.role_service import RoleService
        mysql_conn.init_database()
        with mysql_conn.session_scope() as db:
            added = RoleService.seed_defaults(db)
            logger.info("角色种子数据: 新增 %s 个", added)
    except Exception as e:
        logger.error("启动初始化失败（MySQL 不可用？）: %s", e)
        logger.error("服务仍会启动，但角色/会话相关接口将不可用")
    logger.info("=" * 60)
    yield
    logger.info("服务已停止")


app = FastAPI(
    title="多角色 RAG 角色扮演系统",
    description=(
        "基于 RAG 的角色扮演对话系统。\n\n"
        "- 向量化：BGE-M3（稠密 + 稀疏）\n"
        "- 检索：Milvus 混合检索 + RRF 融合 + BGE-reranker 精排\n"
        "- 记忆：Redis 短期记忆 + Milvus 长期记忆\n"
        "- 角色：律师 / 心理医生 / 金融理财师（人设存 MySQL，可扩展）"
    ),
    version="3.0",
    lifespan=lifespan,
)

for _router in ROUTERS:
    app.include_router(_router)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("未捕获异常 %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500,
                        content={"code": 500, "msg": "服务内部错误: %s" % exc})


@app.get("/", tags=["系统"], summary="服务概览")
async def root():
    return {
        "name": "多角色 RAG 角色扮演系统",
        "version": "3.0",
        "model": settings.LLM_MODEL,
        "docs": "/docs",
        "endpoints": [
            "POST /api/chat/ask",
            "POST /api/chat/stream",
            "GET  /api/roles",
            "GET  /api/sessions",
            "POST /api/ingest/dataset",
            "POST /api/ingest/upload",
            "POST /api/update/document",
            "GET  /api/health",
            "GET  /api/stats",
        ],
    }
