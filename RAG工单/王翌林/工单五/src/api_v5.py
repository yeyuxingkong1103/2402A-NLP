# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-Query 理解优化任务
src/api_v5.py —— 工单五 FastAPI 多轮对话服务（新增文件，不影响 v3/v4）

启动：uvicorn src.api_v5:app --host 0.0.0.0 --port 8005
接口：
  GET  /api/v5/health              健康检查（会话数 + 图像库行数）
  POST /api/v5/chat                多轮对话（session_id 维度过话）
  GET  /api/v5/history/{sid}       获取会话历史
  POST /api/v5/feedback            用户反馈（点赞/点踩/评论）
  POST /api/v5/reset/{sid}         重置会话
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from loguru import logger

from dotenv import load_dotenv
load_dotenv()                                   # 工单五：加载 .env

from src.schemas_v5 import (ChatV5Request, ChatV5Response, FeedbackV5Request,
                            FeedbackV5Response, HealthV5Response,
                            HistoryV5Response)

WORK_ORDER = "人工智能NLP-RAG-Query 理解优化任务"

app = FastAPI(
    title="Query 理解优化（多轮对话）API v5",
    description="工单编号：人工智能NLP-RAG-Query 理解优化任务（工单五）",
    version="5.0.0",
)

_engine = None


def get_engine():
    """工单五：懒加载 ConversationEngine（含 RAGEngineV4）"""
    global _engine
    if _engine is None:
        from src.conversation_engine import ConversationEngine
        _engine = ConversationEngine()
        logger.info("[api_v5] ConversationEngine 已加载")
    return _engine


# ================= 工单五：健康检查 =================
@app.get("/api/v5/health", response_model=HealthV5Response, tags=["工单五"])
async def health():
    """工单五：会话数 + RAG 健康"""
    try:
        stats = get_engine().health()
    except Exception:
        stats = {"sessions": 0, "milvus_images_rows": 0}
    return HealthV5Response(**stats)


# ================= 工单五：多轮对话 =================
@app.post("/api/v5/chat", response_model=ChatV5Response, tags=["工单五"])
async def chat(req: ChatV5Request):
    """工单五：多轮对话（指代消解 + RAGEngineV4 检索问答）"""
    try:
        engine = get_engine()
        if not req.use_rag:
            # 工单五：纯 LLM 模式（不走 RAG，但仍记录会话历史）
            rag = engine._get_rag()
            result = rag.ask_llm(req.question)
            session = engine.get_session(req.session_id)
            session.add_turn(req.question, result.get("answer", ""))
            result.update({
                "session_id": session.session_id,
                "resolved_query": req.question,
                "entity": session.current_entity,
                "doc_id": req.doc_id,
                "is_followup": False,
                "coref_strategy": "pure_llm",
                "conversation_latency_ms": result.get("latency_ms", 0),
            })
            return ChatV5Response(**result)

        result = engine.chat(
            req.question, session_id=req.session_id,
            doc_id=req.doc_id, use_image=req.use_image,
            lang_override=req.lang_override, top_k=req.top_k)
        return ChatV5Response(**result)
    except Exception as e:                      # 工单五：容错
        logger.exception("[api_v5] chat 失败")
        raise HTTPException(status_code=500, detail=f"对话失败: {e}")


# ================= 工单五：会话历史 =================
@app.get("/api/v5/history/{session_id}", response_model=HistoryV5Response,
         tags=["工单五"])
async def get_history(session_id: str):
    """工单五：获取指定会话的完整历史"""
    try:
        h = get_engine().get_history(session_id)
        return HistoryV5Response(
            session_id=h["session_id"], history=h["history"],
            current_entity=h["current_entity"], current_doc=h["current_doc"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"历史查询失败: {e}")


# ================= 工单五：用户反馈 =================
@app.post("/api/v5/feedback", response_model=FeedbackV5Response,
          tags=["工单五"])
async def feedback(req: FeedbackV5Request):
    """工单五：点赞/点踩/评论 → data/feedback_v5/feedback_*.json"""
    try:
        fb_dir = Path("data/feedback_v5")
        fb_dir.mkdir(parents=True, exist_ok=True)
        path = fb_dir / f"feedback_{datetime.now():%Y%m%d}.json"
        records = []
        if path.exists():
            records = json.loads(path.read_text(encoding="utf-8"))
        records.append({
            "ts": datetime.now().isoformat(),
            "session_id": req.session_id,
            "question": req.question,
            "answer": req.answer,
            "rating": req.rating,
            "comment": req.comment,
            "latency_ms": req.latency_ms,
            "work_order": WORK_ORDER,
        })
        path.write_text(json.dumps(records, ensure_ascii=False, indent=1),
                        encoding="utf-8")
        return FeedbackV5Response(ok=True, message=f"反馈已保存: {path}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"反馈保存失败: {e}")


# ================= 工单五：重置会话 =================
@app.post("/api/v5/reset/{session_id}", tags=["工单五"])
async def reset_session(session_id: str):
    """工单五：清空指定会话的历史与实体追踪"""
    get_engine().reset_session(session_id)
    return {"ok": True, "session_id": session_id, "message": "会话已重置"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)
