# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
# 工单01 - 基于PDF文档的问答系统
"""
FastAPI 应用入口。

【为什么单 worker】
8GB 显存里 qwen3:8b(5.2G) + bge-m3(0.66G) 已是紧平衡，多 worker 会各自
持有一份模型句柄并争抢显存，结果是全部变慢甚至 OOM。
并发控制在**应用层**用 asyncio.Semaphore 做（见 api/chat.py），
保证「高并发下稳定运行」这条验收 —— 请求排队而不是把显存打爆。
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api import asr, chat, evaluate, health, ingest, kb
from app.config import settings
from app.core.generator import warmup
from app.core.vectorstore import VectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)
log = logging.getLogger("rag")

STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("工单01 · 基于PDF文档的问答系统 启动")
    log.info("Ollama: %s | LLM: %s | Embed: %s",
             settings.ollama_base_url, settings.llm_model, settings.embed_model)
    log.info("数据目录: %s", settings.data_path)

    ok, msg = VectorStore().health()
    log.info("Milvus: %s", "OK" if ok else f"不可用 —— {msg[:120]}")

    # 预热：把 qwen3 拉进常驻，否则第一个真实请求要吃 ~7s 冷启动
    try:
        await warmup()
        log.info("模型预热完成")
    except Exception as e:  # noqa: BLE001
        log.warning("模型预热失败（不阻断启动）：%s", e)

    yield
    log.info("服务停止")


app = FastAPI(
    title="工单03 · PDF文档的表格解析及检索优化",
    # 工单03 起库里有两份招股意向书，标题里点名任何一家都不对了
    description="人工智能NLP-RAG · 招股意向书问答（兴图新科 / 力源信息，按问题里的公司专名自动路由）",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

for r in (health.router, chat.router, ingest.router, kb.router,
          evaluate.router, asr.router):
    app.include_router(r)


@app.get("/", include_in_schema=False)
async def index():
    return FileResponse(STATIC_DIR / "index.html")


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
