# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
# 【FastAPI后端 · main.py】异步 RESTful API：问答/基线/真实RAGAS评估/反馈/知识库管理
# 编写日期：2026-09-28   修订日期：2026-10-04
import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, Query
from pydantic import BaseModel

import config
import rag_engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动预热：在独立线程加载 Embedding 模型与 BM25 索引，避免首个用户请求冷启动超 3s"""
    import asyncio

    def _warm():
        from embedder import Embedder
        from bm25_retriever import BM25Retriever
        from vector_store import VectorStore
        from llm_client import LLMClient
        import numpy as np

        Embedder()                                   # 触发 bge 模型加载
        BM25Retriever().load()                       # 预载 BM25 索引
        # 预载 FAISS 索引并预热 torch 计算线程（一次真实检索）
        qv = Embedder().encode_one("发行人基本情况")
        VectorStore.search(np.asarray(qv), top_k=3)
        # 预热 LLM 连接池（max_tokens=1 的最小调用，避免首个用户请求握手开销）
        try:
            LLMClient().client.chat.completions.create(
                model=config.LLM_MODEL,
                messages=[{"role": "user", "content": "1"}],
                max_tokens=1,
            )
        except Exception as e:
            print(f"[WARN] LLM 预热失败（不影响运行）: {e}")

    await asyncio.to_thread(_warm)
    # 预热异步 LLM 连接（ask_async 实际使用 AsyncOpenAI 连接池）
    try:
        from llm_client import LLMClient
        acli = LLMClient.async_client()
        await acli.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[{"role": "user", "content": "1"}],
            max_tokens=1,
        )
    except Exception as e:
        print(f"[WARN] 异步 LLM 预热失败（不影响运行）: {e}")
    print("[OK] 启动预热完成：Embedding / BM25 / FAISS / LLM 已就绪")
    yield


app = FastAPI(title="RAG PDF QA API", version="2.0", lifespan=lifespan)


# ---------- 请求/响应模型 ----------
class AskRequest(BaseModel):
    """问答请求"""
    question: str
    top_k: Optional[int] = None
    use_cache: bool = True
    lang: str = "zh"


class EvalItem(BaseModel):
    """评估条目"""
    question: str
    answer: str = ""
    ground_truth: str = ""
    contexts: List[str] = []


class EvalRequest(BaseModel):
    """通用评估请求"""
    items: List[EvalItem]


class FeedbackRequest(BaseModel):
    """用户反馈请求"""
    question: str
    answer: str
    rating: str               # up / down
    comment: str = ""


def _wrap(data, code: int = 0, message: str = "ok"):
    """统一响应包"""
    return {"code": code, "message": message, "data": data}


# ---------- 健康检查 ----------
@app.get("/api/health")
def health():
    """服务健康检查"""
    info = rag_engine.health_check()
    return _wrap(info)


# ---------- RAG 问答 ----------
@app.post("/api/ask")
async def ask_question(req: AskRequest):
    """RAG 问答（异步：混合检索 + LLM 生成）"""
    if not req.question.strip():
        return _wrap(None, code=40001, message="question 不能为空")
    try:
        data = await rag_engine.ask_async(
            req.question, req.top_k, req.use_cache, req.lang
        )
        return _wrap(data)
    except Exception as e:
        return _wrap(None, code=50001, message=f"RAG 问答失败: {e}")


# ---------- 基线问答 ----------
@app.post("/api/ask/baseline")
async def ask_baseline(req: AskRequest):
    """纯 LLM 基线问答（无检索，用于对比分析）"""
    if not req.question.strip():
        return _wrap(None, code=40001, message="question 不能为空")
    try:
        data = await rag_engine.ask_baseline_async(req.question, req.lang)
        return _wrap(data)
    except Exception as e:
        return _wrap(None, code=50001, message=f"基线问答失败: {e}")


# ---------- 真实 RAGAS 评估 ----------
@app.post("/api/evaluate")
async def evaluate(req: EvalRequest):
    """对给定条目执行真实 RAGAS 四指标评估"""
    if not req.items:
        return _wrap(None, code=40001, message="items 不能为空")
    try:
        import evaluator
        records = [it.model_dump() for it in req.items]
        # CPU/LLM 密集型，放线程池避免阻塞事件循环
        import asyncio
        data = await asyncio.to_thread(evaluator.run_ragas, records)
        return _wrap(data)
    except Exception as e:
        return _wrap(None, code=50002, message=f"RAGAS 评估失败: {e}")


@app.post("/api/evaluate/workorder")
async def evaluate_workorder():
    """一键对工单 10 题完成 RAG 问答 + 真实 RAGAS 评估（耗时较长，约数分钟）"""
    try:
        import evaluator
        import asyncio
        data = await asyncio.to_thread(evaluator.evaluate_workorder)
        return _wrap(data)
    except Exception as e:
        return _wrap(None, code=50002, message=f"工单评估失败: {e}")


# ---------- 用户反馈 ----------
@app.post("/api/feedback")
def feedback(req: FeedbackRequest):
    """提交答案反馈（点赞/点踩 + 文字）"""
    if req.rating not in ("up", "down"):
        return _wrap(None, code=40002, message="rating 必须为 up/down")
    try:
        rec = rag_engine.save_feedback(
            req.question, req.answer, req.rating, req.comment
        )
        return _wrap(rec)
    except Exception as e:
        return _wrap(None, code=50003, message=f"反馈保存失败: {e}")


@app.get("/api/feedback")
def get_feedback(limit: int = Query(100, ge=1, le=1000)):
    """查看反馈列表"""
    return _wrap({"items": rag_engine.list_feedback(limit)})


# ---------- 知识库管理 ----------
@app.get("/api/documents")
def list_documents():
    """列出全部已入库文档与后端信息"""
    return _wrap(rag_engine.list_documents())


@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...),
                     doc_id: Optional[str] = Query(None)):
    """上传 PDF 并执行入库流水线"""
    if not file.filename.lower().endswith(".pdf"):
        return _wrap(None, code=42201, message="仅支持 PDF 文件")
    doc_id = doc_id or Path(file.filename).stem
    save_path = config.UPLOAD_DIR / file.filename
    with save_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    import asyncio
    data = await asyncio.to_thread(
        rag_engine.ingest, str(save_path), doc_id, file.filename
    )
    return _wrap(data)


@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: str):
    """删除指定文档（向量 + BM25 + 元信息）"""
    return _wrap(rag_engine.delete_document(doc_id))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.API_HOST, port=config.API_PORT)

# ====================================================================
# 技术备注：
# 1. RAG：/api/ask 走完整 RAG；/api/ask/baseline 用于工单要求的对比分析；
#    /api/evaluate* 提供真实 RAGAS 评估结果。
# 2. 全异步路由 + 线程池隔离 CPU 密集任务，单机并发吞吐显著提升。
# 3. Transformer：问答与评估背后均为 Transformer 模型。
# 4. Fine-tuning：API 层与模型实现解耦，微调后只需重新入库。
# ====================================================================
