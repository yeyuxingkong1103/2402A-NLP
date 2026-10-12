# -*- coding: utf-8 -*-
"""
工单十五：FastAPI 服务
"""
import os, json
from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI
from pydantic import BaseModel
from src.config import WORK_ORDER, WORK_ORDER_NO, APP_HOST, APP_PORT
from src.rag_engine import RAGEngine
from loguru import logger

app = FastAPI(title=f"{WORK_ORDER_NO} 跨模态检索 RAG", version="1.0")
_engine: RAGEngine = None


def get_engine() -> RAGEngine:
    global _engine
    if _engine is None:
        _engine = RAGEngine(top_k=5)
        mapping_json = os.path.join(os.path.dirname(__file__), "..", "data", "kb_mapping.json")
        mapping_json = os.path.abspath(mapping_json)
        if os.path.isfile(mapping_json):
            with open(mapping_json, encoding="utf-8") as f:
                m = json.load(f)
            _engine.set_page_images({int(k): v for k, v in m.get("page_images", {}).items()})
            _engine.set_page_texts({int(k): v for k, v in m.get("page_texts", {}).items()})
        logger.info("RAGEngine 就绪")
    return _engine


class Query(BaseModel):
    question: str


@app.get("/")
def root():
    return {"work_order": WORK_ORDER_NO, "task": WORK_ORDER, "status": "ok"}


@app.post("/ask")
def ask(q: Query):
    engine = get_engine()
    return engine.ask(q.question)


@app.get("/health")
def health():
    engine = get_engine()
    stats = engine.retriever.vs.get_stats()
    return {"status": "ok", "milvus": stats}
