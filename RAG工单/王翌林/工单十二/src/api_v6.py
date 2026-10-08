# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-混合检索任务
src/api_v6.py —— 工单六 可配置混合检索 FastAPI 服务（新增文件，不影响 v3/v4/v5）

启动：uvicorn src.api_v6:app --host 0.0.0.0 --port 8006
接口：
  GET  /api/v6/health              健康检查（图像库行数 + 全文索引文档数）
  GET  /api/v6/config              检索策略可选项与预设
  POST /api/v6/ask                 可配置混合检索问答（vector/fulltext/hybrid）
  POST /api/v6/chat                多轮对话（沿用工单五）+ 检索策略
  GET  /api/v6/history/{sid}       会话历史
  POST /api/v6/feedback            用户反馈（供自适应重排器学习）
  POST /api/v6/reset/{sid}         重置会话
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from loguru import logger

from dotenv import load_dotenv
load_dotenv()                                   # 工单六：加载 .env

from src.retrieval.retrieval_config import (
    PRESETS, VALID_FIELDS, VALID_FUSIONS, VALID_MATCHES, VALID_MODES,
    VALID_RERANKERS, RetrievalConfig,
)
from src.schemas_v6 import (AskV6Request, AskV6Response, ChatV6Request,
                            ConfigV6Response, HealthV6Response)

WORK_ORDER = "人工智能NLP-RAG-混合检索任务"

app = FastAPI(
    title="混合检索策略 API v6",
    description="工单编号：人工智能NLP-RAG-混合检索任务（工单六）",
    version="6.0.0",
)

_engine = None
_conv = None


def get_engine():
    """工单六：懒加载 RAGEngineV6（可配置混合检索 + 工单四图文表链路）"""
    global _engine
    if _engine is None:
        from src.rag_engine_v6 import RAGEngineV6
        _engine = RAGEngineV6(top_k=5)
        logger.info("[api_v6] RAGEngineV6 已加载")
    return _engine


def get_conv():
    """工单六：多轮对话引擎（用工单五 ConversationEngine 包装 v6 RAG）"""
    global _conv
    if _conv is None:
        from src.conversation_engine import ConversationEngine
        _conv = ConversationEngine(rag_engine=get_engine())
        logger.info("[api_v6] ConversationEngine(v6) 已加载")
    return _conv


def _retrieval_kwargs(req_retrieval) -> Dict[str, Any]:
    if req_retrieval is None:
        return {}
    data = req_retrieval.model_dump(exclude_none=True)
    return {"retrieval_config": data} if data else {}


# ================= 工单六：健康检查 =================
@app.get("/api/v6/health", response_model=HealthV6Response, tags=["工单六"])
async def health():
    try:
        stats = get_engine().health()
    except Exception:
        stats = {"milvus_images_rows": 0, "fulltext_docs": 0}
    return HealthV6Response(**stats)


# ================= 工单六：策略配置 =================
@app.get("/api/v6/config", response_model=ConfigV6Response, tags=["工单六"])
async def get_config():
    """工单六：返回三种模式/两种融合/三种重排器/四种匹配方式与预设"""
    return ConfigV6Response(
        modes=list(VALID_MODES), fusions=list(VALID_FUSIONS),
        rerankers=list(VALID_RERANKERS), matches=list(VALID_MATCHES),
        fields=list(VALID_FIELDS), presets=PRESETS,
        defaults=RetrievalConfig().to_dict())


# ================= 工单六：可配置混合检索问答 =================
@app.post("/api/v6/ask", response_model=AskV6Response, tags=["工单六"])
async def ask(req: AskV6Request):
    try:
        engine = get_engine()
        if not req.use_rag:
            return AskV6Response(**engine.ask_llm(req.question))
        result = engine.ask(
            req.question, doc_id=req.doc_id, company=req.company,
            lang_override=req.lang_override, **_retrieval_kwargs(req.retrieval))
        return AskV6Response(**result)
    except Exception as e:
        logger.exception("[api_v6] ask 失败")
        raise HTTPException(status_code=500, detail=f"问答失败: {e}")


# ================= 工单六：多轮对话（沿用工单五） =================
@app.post("/api/v6/chat", response_model=AskV6Response, tags=["工单六"])
async def chat_turn(req: ChatV6Request):
    try:
        conv = get_conv()
        kw = _retrieval_kwargs(req.retrieval)
        result = conv.chat(req.question, session_id=req.session_id,
                           doc_id=req.doc_id, use_image=req.use_image, **kw)
        return AskV6Response(**result)
    except Exception as e:
        logger.exception("[api_v6] chat 失败")
        raise HTTPException(status_code=500, detail=f"对话失败: {e}")


@app.get("/api/v6/history/{session_id}", tags=["工单六"])
async def get_history(session_id: str):
    try:
        return get_conv().get_history(session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"历史查询失败: {e}")


@app.post("/api/v6/reset/{session_id}", tags=["工单六"])
async def reset_session(session_id: str):
    get_conv().reset_session(session_id)
    return {"ok": True, "session_id": session_id, "message": "会话已重置"}


# ================= 工单六：用户反馈（自适应重排器的学习数据） =================
@app.post("/api/v6/feedback", tags=["工单六"])
async def feedback(payload: dict):
    """工单六：点赞/点踩/评论 → data/feedback_v6/（AdaptiveReranker 读取）"""
    try:
        fb_dir = Path("data/feedback_v6")
        fb_dir.mkdir(parents=True, exist_ok=True)
        path = fb_dir / f"feedback_{datetime.now():%Y%m%d}.json"
        records = []
        if path.exists():
            records = json.loads(path.read_text(encoding="utf-8"))
        payload["ts"] = datetime.now().isoformat()
        payload["work_order"] = WORK_ORDER
        records.append(payload)
        path.write_text(json.dumps(records, ensure_ascii=False, indent=1),
                        encoding="utf-8")
        return {"ok": True, "message": f"反馈已保存: {path}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"反馈保存失败: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8006)
