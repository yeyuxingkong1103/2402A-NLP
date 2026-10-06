# -*- coding: utf-8 -*-
"""
工单：人工智能NLP-RAG-基于PDF文档的问答系统
src/api.py — FastAPI 后端
"""
import os, time
from contextlib import asynccontextmanager
from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI, HTTPException
from loguru import logger
from src.schemas import AskRequest, AskResponse, FeedbackRequest, FeedbackResponse, HealthResponse, ReferenceItem, StatsResponse

_START_TIME = time.time()

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 RAG API 启动中...")
    try:
        from src.db import test_connection
        ver = test_connection()
        app.state.mysql_status = f"ok ({ver})" if ver else "failed"
    except Exception as e: app.state.mysql_status = f"failed: {e}"
    try:
        from src.vector_store import VectorStore
        vs = VectorStore(); stats = vs.get_stats()
        app.state.milvus_status = f"ok ({stats.get('mode','?')})"
        app.state.vector_store = vs
    except Exception as e: app.state.milvus_status = f"failed: {e}"
    try:
        from src.rag_engine import RAGEngine
        engine = RAGEngine()
        app.state.rag_engine = engine; app.state.retriever = engine.retriever
    except Exception as e: app.state.milvus_status = f"failed: {e}"
    logger.info(f"✅ RAG API 就绪 | mysql={app.state.mysql_status} | milvus={app.state.milvus_status}")
    yield
    logger.info("🛑 RAG API 关闭")

app = FastAPI(title="RAG PDF QA API", version="1.0.0", lifespan=lifespan,
              description="基于 PDF 招股说明书的 RAG 问答系统 API （工单：人工智能NLP-RAG-基于PDF文档的问答系统）")

@app.get("/api/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", mysql=getattr(app.state, "mysql_status", "unknown"),
                          milvus=getattr(app.state, "milvus_status", "unknown"),
                          embedding_model=os.getenv("EMBEDDING_MODEL", ""),
                          llm_model=os.getenv("DEEPSEEK_MODEL", ""),
                          uptime_seconds=round(time.time() - _START_TIME, 1))

@app.get("/api/stats", response_model=StatsResponse)
def stats():
    out = StatsResponse()
    try:
        from src.db import get_session
        with get_session() as sess:
            from src.models import Document, Chunk, QALog, Feedback
            out.total_documents = sess.query(Document).count()
            out.total_chunks = sess.query(Chunk).count()
            out.total_qa_logs = sess.query(QALog).count()
            out.total_feedback = sess.query(Feedback).count()
    except Exception as e: logger.warning(f"stats mysql fail: {e}")
    try:
        vs = getattr(app.state, "vector_store", None)
        if vs: out.collection_entities = vs.get_stats().get("num_entities")
    except Exception as e: logger.warning(f"stats milvus fail: {e}")
    return out

def _log_qa(question, rag_answer, llm_answer, rag_result=None, llm_result=None):
    """工单：人工智能NLP-RAG-基于PDF文档的问答系统 —— 记录问答日志，返回 qa_log_id"""
    try:
        from src.db import get_session
        from src.models import QALog
        with get_session() as sess:
            log = QALog(question=question,
                        query_rewrite=(rag_result or {}).get("query_understanding", {}).get("rewrite") if rag_result else None,
                        rag_answer=rag_answer, llm_answer=llm_answer,
                        retrieved_chunks=(rag_result or {}).get("references") if rag_result else None,
                        latency_ms=(rag_result or {}).get("latency_ms", 0.0) if rag_result else 0.0,
                        llm_latency_ms=(llm_result or {}).get("latency_ms", 0.0) if llm_result else 0.0,
                        token_usage=(rag_result or {}).get("token_usage") if rag_result else None)
            sess.add(log); sess.commit(); sess.refresh(log)
            return log.id
    except Exception as e:
        logger.warning(f"QALog 记录失败(不影响回答): {e}"); return None

@app.post("/api/ask", response_model=AskResponse)
def ask(req: AskRequest):
    engine = getattr(app.state, "rag_engine", None)
    if engine is None: raise HTTPException(status_code=503, detail="RAG Engine 未就绪")
    try:
        if not req.use_rag:
            r = engine.ask_llm(req.question)
            qa_id = _log_qa(req.question, None, r["answer"], llm_result=r)
            return AskResponse(mode=r["mode"], answer=r["answer"], qa_log_id=qa_id, references=[],
                                latency_ms=r["latency_ms"], token_usage=r.get("token_usage", {}),
                                llm_answer=r["answer"], llm_latency_ms=r["latency_ms"])
        # 工单二：优化链路（语义缓存+父子块+多语言，人工智能NLP-RAG-基于PDF文档的问答系统优化）
        lang = getattr(req, "lang", None)  # 前端语言切换透传（None=自动检测）
        chain = getattr(req, "chain", None)  # 工单二：baseline=工单一基线（前端对比用）
        if chain == "baseline":
            r = engine.ask_rag(req.question)
        else:
            r = engine.ask_optimized(req.question, lang_override=lang)
        refs = [ReferenceItem(**ref) for ref in r.get("references", [])]
        qa_id = _log_qa(req.question, r["answer"], None, rag_result=r)
        return AskResponse(mode=r["mode"], answer=r["answer"], qa_log_id=qa_id, references=refs,
                            latency_ms=r["latency_ms"], token_usage=r.get("token_usage", {}),
                            query_understanding=r.get("query_understanding"), breakdown=r.get("breakdown"),
                            # 工单二：透传语义缓存命中标记给前端耗时面板（人工智能NLP-RAG-基于PDF文档的问答系统优化）
                            cache_hit=r.get("cache_hit"),
                            rag_answer=r["answer"], rag_latency_ms=r["latency_ms"])
    except Exception as e:
        logger.error(f"/api/ask 失败: {e}"); raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/feedback", response_model=FeedbackResponse)
def feedback(req: FeedbackRequest):
    try:
        from src.db import get_session
        from src.models import Feedback as FB
        with get_session() as sess:
            fb = FB(qa_log_id=req.qa_log_id, rating=req.rating, is_correct=req.is_correct, comment=req.comment)
            sess.add(fb); sess.commit(); sess.refresh(fb)
        return FeedbackResponse(id=fb.id)
    except Exception as e:
        logger.error(f"/api/feedback 失败: {e}"); raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/questions")
def example_questions():
    return [
        {"id": 1, "text": "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？"},
        {"id": 2, "text": "公司的主要风险因素有哪些？"},
        {"id": 3, "text": "武汉兴图新科电子股份有限公司法定代表人是谁？"},
        {"id": 4, "text": "公司本次发行的股票数量和每股面值是多少？"},
        {"id": 5, "text": "报告期内公司的净利润分别是多少？"},
    ]

# ========== 工单：人工智能NLP-RAG-基于PDF文档的问答系统 —— SSE 流式端点 ==========
@app.post("/api/ask_stream")
async def ask_stream(req: AskRequest):
    """工单：人工智能NLP-RAG-基于PDF文档的问答系统 —— SSE 流式问答"""
    from fastapi.responses import StreamingResponse
    import json as _json

    engine = getattr(app.state, "rag_engine", None)
    if engine is None: raise HTTPException(status_code=503, detail="RAG Engine 未就绪")

    async def generate():
        # Step 1: 查询理解
        qu = engine.retriever.retrieve if hasattr(engine.retriever, 'retrieve') else None
        yield f"data: {_json.dumps({'type': 'status', 'msg': '正在检索...'}, ensure_ascii=False)}\n\n"

        # Step 2: 检索
        t0 = time.time()
        try:
            rewritten = req.question
            chunks = engine.retriever.retrieve(rewritten, top_k=req.top_k)
            ref_list = [{"page": c.get("page"), "chunk_id": c.get("chunk_id"),
                         "score": c.get("score"), "preview": c.get("content", "")[:80]} for c in chunks]
            yield f"data: {_json.dumps({'type': 'retrieved', 'count': len(chunks), 'latency_ms': round((time.time()-t0)*1000,1), 'references': ref_list}, ensure_ascii=False)}\n\n"
        except Exception as e:
            yield f"data: {_json.dumps({'type': 'error', 'msg': str(e)}, ensure_ascii=False)}\n\n"
            return

        # Step 3: LLM 生成（模拟流式 —— deepseek API 不支持真正的 SSE 但我们逐字推送）
        from src.llm_client import chat
        from src.retriever import Retriever
        if chunks:
            context = engine._truncate_context(chunks)
            from src.rag_engine import RAG_PROMPT_TEMPLATE, RAG_SYSTEM
            prompt = RAG_PROMPT_TEMPLATE.format(context=context, query=req.question)
            llm_result = chat(messages=[{"role": "system", "content": RAG_SYSTEM},
                                       {"role": "user", "content": prompt}])
        else:
            llm_result = chat(messages=[{"role": "system", "content": "你是一名投资分析师。"},
                                       {"role": "user", "content": req.question}])

        # 逐字推送（工单：模拟流式输出）
        answer = llm_result.get("content", "")
        for i in range(0, len(answer), 3):
            chunk_text = answer[i:i+3]
            yield f"data: {_json.dumps({'type': 'token', 'text': chunk_text}, ensure_ascii=False)}\n\n"

        # 结束
        total_ms = round((time.time() - t0) * 1000, 1)
        yield f"data: {_json.dumps({'type': 'done', 'latency_ms': total_ms, 'token_usage': llm_result.get('token_usage', {})}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api:app", host=os.getenv("APP_HOST", "0.0.0.0"),
                port=int(os.getenv("APP_PORT", "8001")), reload=False)
