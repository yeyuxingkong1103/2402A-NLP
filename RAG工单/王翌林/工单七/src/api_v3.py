# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
src/api_v3.py —— 工单三 FastAPI 接口（端口 8003）

启动：uvicorn src.api_v3:app --host 0.0.0.0 --port 8003 --reload

接口：
  GET  /api/v3/health          健康检查
  POST /api/v3/ask             表格感知 RAG 问答
  GET  /api/v3/tables/{doc_id} 获取文档所有表格
  POST /api/v3/upload          上传新 PDF 并入库
  POST /api/v3/evaluate        运行 14 题评估
"""
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# 工单三：项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 工单三：显存优化环境变量
os.environ.setdefault("RAG_EMBED_DEVICE", "cuda")
os.environ.setdefault("RAG_EMBED_BATCH_SIZE", "8")
os.environ.setdefault("RAG_EMBED_MAX_SEQ", "512")

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
import shutil

from src.schemas_v3 import (
    AskV3Request, AskV3Response, ReferenceV3,
    TableV3Response, UploadResponse, EvaluateRequest,
    EvaluateResponse, HealthResponse,
)

app = FastAPI(
    title="PDF 表格解析与检索优化 API（工单三）",
    description="工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= 引擎懒加载 =================
_engine = None


def get_engine():
    """工单三：懒加载 RAGEngineV3"""
    global _engine
    if _engine is None:
        from src.rag_engine_v3 import RAGEngineV3
        _engine = RAGEngineV3(top_k=5, use_rerank=True)
        logger.info("[api_v3] RAGEngineV3 已加载")
    return _engine


# ================= 1. 健康检查 =================
@app.get("/api/v3/health", response_model=HealthResponse,
         tags=["工单三"])
async def health():
    """工单三：健康检查"""
    return HealthResponse(
        status="ok", version="v3",
        table_collection="rag_tables",
        text_collection="rag_chunks",
    )


# ================= 2. 表格感知 RAG 问答 =================
@app.post("/api/v3/ask", response_model=AskV3Response,
          tags=["工单三"])
async def ask(req: AskV3Request):
    """工单三：表格感知 RAG 问答

    - use_rag=True: 路由 → 表格+文本检索 → rerank → LLM
    - use_rag=False: 纯 LLM 模式（对照基线）
    - use_table=False: 仅文本检索（模拟工单二）
    """
    engine = get_engine()

    if not req.use_rag:
        # 纯 LLM 模式
        result = engine.ask_llm(req.question)
        return AskV3Response(
            mode="pure_llm", question=req.question,
            answer=result["answer"], references=[],
            latency_ms=result["latency_ms"],
        )

    # 工单三：RAG 模式（含中英文翻译路由）
    from src.multilingual_v3 import bilingual_ask_rag
    result = bilingual_ask_rag(
        engine, req.question,
        doc_id=req.doc_id, top_k=req.top_k,
        lang_override=req.lang,
    )

    # 组装引用
    refs = []
    for ref in result.get("references", []):
        refs.append(ReferenceV3(
            type=ref.get("type", "text"),
            ref_id=ref.get("ref_id", ""),
            page=ref.get("page"),
            doc_id=ref.get("doc_id"),
            table_id=ref.get("table_id"),
            score=ref.get("score", 0.0),
            preview=ref.get("preview", ""),
        ))

    return AskV3Response(
        mode="rag_v3",
        question=req.question,
        answer=result.get("answer", ""),
        references=refs,
        latency_ms=result.get("total_ms",
                              result.get("latency_ms", 0)),
        breakdown=result.get("breakdown", {}),
        route=result.get("route", {}),
        retrieved_text_chunks=result.get("retrieved_text_chunks", []),
        retrieved_tables=result.get("retrieved_tables", []),
        lang=result.get("lang", "zh"),
        translated=result.get("translated", False),
    )


# ================= 3. 获取文档表格 =================
@app.get("/api/v3/tables/{doc_id}", response_model=TableV3Response,
         tags=["工单三"])
async def get_tables(doc_id: str):
    """工单三：返回指定文档的所有解析表格"""
    table_file = PROJECT_ROOT / "data" / "tables" / f"{doc_id}_tables.json"
    if not table_file.exists():
        raise HTTPException(404, f"文档 {doc_id} 的表格文件不存在")
    data = json.loads(table_file.read_text(encoding="utf-8"))
    tables = data.get("tables", [])
    return TableV3Response(
        doc_id=doc_id, total=len(tables), tables=tables[:100],
    )


# ================= 4. 上传新 PDF =================
@app.post("/api/v3/upload", response_model=UploadResponse,
          tags=["工单三"])
async def upload_pdf(pdf: UploadFile = File(...),
                     doc_name: Optional[str] = Query(None),
                     company: Optional[str] = Query(None)):
    """工单三：上传新 PDF → 解析 → 入库（文本+表格）

    流程：保存文件 → pdf_parser_v3 解析 → 向量化 → 入库
    """
    if not pdf.filename or not pdf.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "请上传 PDF 文件")

    # 保存到附件目录
    upload_dir = PROJECT_ROOT / "附件"
    upload_dir.mkdir(parents=True, exist_ok=True)
    save_path = upload_dir / pdf.filename
    with open(save_path, "wb") as f:
        shutil.copyfileobj(pdf.file, f)

    # 解析入库
    try:
        from src.pdf_parser_v3 import parse_pdf_v3
        from src.table_parser.table_embedding import embed_table_chunks
        from src.table_parser.table_store import TableStore
        from src.embedding import get_embedder
        from src.vector_store import VectorStore

        doc_name = doc_name or Path(pdf.filename).stem
        company = company or ""

        # 文本+表格解析
        parsed = parse_pdf_v3(str(save_path), doc_name=doc_name,
                              company=company)
        text_chunks = parsed.get("text_chunks", [])
        table_chunks = parsed.get("table_chunks", [])

        # 文本入库
        embedder = get_embedder()
        texts = [ch.get("text", "") for ch in text_chunks]
        if texts:
            vectors = embedder.encode(texts, batch_size=8,
                                      show_progress_bar=False)
            vs = VectorStore()
            vs.ensure_collection()
            vs.insert_chunks(text_chunks, vectors)
            vs.close()

        # 表格入库
        if table_chunks:
            table_vectors = embed_table_chunks(table_chunks)
            ts = TableStore()
            ts.insert_tables(table_chunks)
            ts.close()

        # 保存 JSON
        tables_file = PROJECT_ROOT / "data" / "tables" / f"{doc_name}_tables.json"
        tables_file.parent.mkdir(parents=True, exist_ok=True)
        parsed_file = PROJECT_ROOT / "data" / "parsed_v3" / f"{doc_name}_text.json"
        parsed_file.parent.mkdir(parents=True, exist_ok=True)

        import hashlib
        doc_id = hashlib.sha1(
            f"{save_path}+{pdf.filename}".encode()
        ).hexdigest()[:16]

        logger.info(f"[api_v3] 上传成功: {pdf.filename}, "
                    f"text={len(text_chunks)}, table={len(table_chunks)}")

        return UploadResponse(
            success=True, doc_name=doc_name, doc_id=doc_name,
            text_chunks=len(text_chunks),
            table_chunks=len(table_chunks),
            message=f"上传成功: {len(text_chunks)} 文本块, "
                    f"{len(table_chunks)} 表格块",
        )
    except Exception as e:
        logger.error(f"[api_v3] 上传失败: {e}")
        raise HTTPException(500, f"解析入库失败: {e}")


# ================= 5. 14 题评估 =================
EVAL_QUESTIONS = [
    {"id": 1, "question": "武汉力源信息技术股份有限公司本次发行股数是多少，占发行后总股本的比例是多少？", "doc_id": "招股说明书2"},
    {"id": 2, "question": "武汉力源信息技术股份有限公司本次募集资金拟投资哪些项目？", "doc_id": "招股说明书2"},
    {"id": 3, "question": "与武汉力源信息技术股份有限公司存在控制关系的关联方是谁，持股比例和本公司关系是什么？", "doc_id": "招股说明书2"},
    {"id": 4, "question": "与武汉力源信息技术股份有限公司不存在控制关系的关联方企业有哪些？", "doc_id": "招股说明书2"},
    {"id": 260, "question": "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？", "doc_id": "招股说明书1"},
    {"id": 95, "question": "武汉兴图新科电子股份有限公司参与制定了哪个技术标准？", "doc_id": "招股说明书1"},
    {"id": 33, "question": "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入占主营业务收入的比重分别是多少？", "doc_id": "招股说明书1"},
    {"id": 34, "question": "根据武汉兴图新科电子股份有限公司招股意向书，电子信息行业的上游涉及哪些企业？", "doc_id": "招股说明书1"},
    {"id": 957, "question": "武汉兴图新科电子股份有限公司在哪个领域已经成为重要供应商？", "doc_id": "招股说明书1"},
    {"id": 793, "question": "根据武汉兴图新科电子股份有限公司招股意向书，电子信息行业的下游主要包括哪些行业？", "doc_id": "招股说明书1"},
    {"id": 795, "question": "武汉兴图新科电子股份有限公司参与的哪个工程荣获了国家科技进步一等奖？", "doc_id": "招股说明书1"},
    {"id": 543, "question": "武汉兴图新科电子股份有限公司注册资本是多少？", "doc_id": "招股说明书1"},
    {"id": 531, "question": "武汉兴图新科电子股份有限公司法定代表人是谁？", "doc_id": "招股说明书1"},
    {"id": 207, "question": "武汉兴图新科电子股份有限公司计划使用本次发行募集资金的多少用于补充流动资金？", "doc_id": "招股说明书1"},
]


@app.post("/api/v3/evaluate", response_model=EvaluateResponse,
          tags=["工单三"])
async def evaluate(req: EvaluateRequest):
    """工单三：运行 14 题评估

    对每个问题分别跑 RAG 模式和纯 LLM 模式，计算准确率/响应时间。
    """
    engine = get_engine()
    questions = req.questions or EVAL_QUESTIONS
    results = []

    for q in questions:
        qid = q.get("id", "?")
        question = q["question"]
        doc_id = q.get("doc_id")

        # RAG 模式
        rag_result = engine.ask_rag(question, doc_id=doc_id, top_k=5)
        # 纯 LLM 模式
        llm_result = engine.ask_llm(question)

        results.append({
            "id": qid,
            "question": question,
            "doc_id": doc_id,
            "rag_v3": {
                "answer": rag_result.get("answer", "")[:300],
                "latency_ms": rag_result.get("latency_ms", 0),
                "route": rag_result.get("route", {}).get("route"),
                "text_chunks": len(rag_result.get("retrieved_text_chunks", [])),
                "table_chunks": len(rag_result.get("retrieved_tables", [])),
            },
            "pure_llm": {
                "answer": llm_result.get("answer", "")[:300],
                "latency_ms": llm_result.get("latency_ms", 0),
            },
        })

    # 汇总
    rag_latencies = [r["rag_v3"]["latency_ms"] for r in results]
    llm_latencies = [r["pure_llm"]["latency_ms"] for r in results]
    summary = {
        "total": len(results),
        "rag_v3_avg_latency_ms": sum(rag_latencies) / len(rag_latencies) if rag_latencies else 0,
        "pure_llm_avg_latency_ms": sum(llm_latencies) / len(llm_latencies) if llm_latencies else 0,
        "rag_v3_table_hits": sum(r["rag_v3"]["table_chunks"] > 0 for r in results),
    }

    return EvaluateResponse(
        total_questions=len(results),
        results=results,
        summary=summary,
    )


# ================= 工单二接口兼容（保留） =================
@app.get("/api/health", tags=["工单二兼容"])
async def health_v2():
    """工单二健康检查（兼容保留）"""
    return {"status": "ok", "version": "v2+v3"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
