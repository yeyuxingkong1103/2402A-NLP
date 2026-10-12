# -*- coding: utf-8 -*-
"""
工单十四：FastAPI 服务
"""
import os
import json
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional
from loguru import logger
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.config import APP_HOST, APP_PORT, WORK_ORDER_NO
from src.rag_engine import RAGEngine

_engine: Optional[RAGEngine] = None


def get_engine() -> RAGEngine:
    global _engine
    if _engine is None:
        _engine = RAGEngine(top_k=5)
        # 加载页面映射
        mapping_path = "./data/kb_mapping.json"
        if os.path.exists(mapping_path):
            with open(mapping_path, "r", encoding="utf-8") as f:
                mapping = json.load(f)
            _engine.set_page_images({int(k): v for k, v in mapping.get("page_images", {}).items()})
            _engine.set_page_texts({int(k): v for k, v in mapping.get("page_texts", {}).items()})
            logger.info("知识库映射已加载")
    return _engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"{WORK_ORDER_NO} API 启动中...")
    get_engine()
    yield
    logger.info(f"{WORK_ORDER_NO} API 关闭")


app = FastAPI(title=f"{WORK_ORDER_NO} 工业PDF多模态RAG", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class AskRequest(BaseModel):
    query: str


class AskResponse(BaseModel):
    answer: str
    mode: str
    latency_ms: float
    route: Dict[str, Any]
    references: List[Dict] = []
    image_path: Optional[str] = None
    image_desc: Optional[str] = None
    token_usage: Dict = {}


@app.get("/health")
def health():
    stats = get_engine().retriever.vs.get_stats()
    return {"status": "ok", "work_order": WORK_ORDER_NO, "milvus": stats}


@app.post("/api/ask", response_model=AskResponse)
def ask(req: AskRequest):
    engine = get_engine()
    result = engine.ask(req.query)
    return AskResponse(**result)


@app.get("/api/questions")
def get_questions():
    """返回工单指定的6个测试问题"""
    return {
        "questions": [
            {"id": 1, "question": "根据文本信息，该静电除尘器的发明人是？", "type": "text",
             "options": ["A. P·吉特勒", "B. 未知", "C. 静电除尘器研究小组", "D. 该专利未提及发明人"]},
            {"id": 2, "question": "根据文本信息，以下哪个描述符合该静电除尘器的特征？", "type": "text",
             "options": ["A. 具有平行于外壳主轴线的垂直方向的片状沉积电极",
                         "B. 具有管状入口和出口，它们分别由3种不同圆锥形部分所构成",
                         "C. 管状入口具有单个圆锥形部分，达到外壳直径的80至95%，剩余部分采用台阶形式",
                         "D. 主要用于液体的除尘"]},
            {"id": 3, "question": "在文件中第7页的图片中，部件4相对于部件5在图片中的位置关系是？", "type": "image",
             "options": ["A. 部件4位于部件5的左侧", "B. 部件4位于部件5的右侧",
                         "C. 部件4位于部件5的上方", "D. 部件4位于部件5的下方"]},
            {"id": 4, "question": "在文件中第7页的图片中，尺寸X1，X2，X3分别代表什么部件的间隔距离？", "type": "image",
             "options": ["A. 配气带孔盘6，6'，6\"之间的间隔距离",
                         "B. 外壳2与圆锥形部分10之间的间隔距离",
                         "C. 管状入口1与管状出口9之间的间隔距离",
                         "D. 外壳中心轴线3与外壳2之间的间隔距离"]},
            {"id": 5, "question": "根据文件中第7页图示，气流方向(7)首先经过哪个部件？紧接着会经过哪个部件？", "type": "image",
             "options": ["A. 先经过部件4，再经过部件5", "B. 先经过部件10，再经过部件6\"",
                         "C. 先经过部件6\"，再经过部件6'", "D. 先经过部件6，再经过部件4"]},
            {"id": 6, "question": "根据文件中第7页图示，如果已知外壳直径D，那么h1和h2的尺寸可以用来计算什么？", "type": "image",
             "options": ["A. 计算气流速度", "B. 确定配气带孔盘6，6'，6\"的位置",
                         "C. 计算除尘效率", "D. 确定外壳2的材料"]},
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=APP_HOST, port=APP_PORT)
