# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-图像内容解析及检索优化
src/api_v4.py —— 工单四 FastAPI 服务（新增文件，不影响 api_v3）

启动：uvicorn src.api_v4:app --host 0.0.0.0 --port 8004 --reload
接口：
  GET  /api/v4/health             健康检查（rag_images 行数）
  POST /api/v4/ask                图文表融合问答
  GET  /api/v4/images/{doc_id}    某册已入库图像列表
  POST /api/v4/upload             上传 PDF → 图像提取/解析/入库
  POST /api/v4/evaluate           批量评估（16 题覆盖）
"""
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from loguru import logger

from dotenv import load_dotenv
load_dotenv()                                   # 工单四：加载 .env（DEEPSEEK_API_KEY 等）

from src.schemas_v4 import (AskV4Request, AskV4Response, EvalItem,
                            EvaluateV4Request, EvaluateV4Response,
                            HealthV4Response, ImageItem, ImagesV4Response,
                            UploadV4Response)

WORK_ORDER = "人工智能NLP-RAG-图像内容解析及检索优化"

app = FastAPI(
    title="PDF文档的图像内容解析及检索优化 API v4",
    description="工单编号：人工智能NLP-RAG-图像内容解析及检索优化（工单四）",
    version="4.0.0",
)

_engine = None


def get_engine():
    """工单四：懒加载 RAGEngineV4（复用工单三全链路 + 新增图像三路）"""
    global _engine
    if _engine is None:
        from src.rag_engine_v4 import RAGEngineV4
        _engine = RAGEngineV4(top_k=5)
        logger.info("[api_v4] RAGEngineV4 已加载")
    return _engine


# ================= 工单四：健康检查 =================
@app.get("/api/v4/health", response_model=HealthV4Response, tags=["工单四"])
async def health():
    """工单四：服务健康 + 图像库规模"""
    try:
        stats = get_engine().health()
    except Exception:
        stats = {"milvus_images_rows": 0}
    return HealthV4Response(**stats)


# ================= 工单四：融合问答 =================
@app.post("/api/v4/ask", response_model=AskV4Response, tags=["工单四"])
async def ask(req: AskV4Request):
    """工单四：文本+表格+图像三路检索 → 融合上下文 → LLM"""
    try:
        engine = get_engine()
        if not req.use_rag:
            return AskV4Response(**engine.ask_llm(req.question))
        result = engine.ask(req.question, doc_id=req.doc_id,
                            company=req.company, use_image=req.use_image,
                            lang_override=req.lang_override, top_k=req.top_k)
        return AskV4Response(**result)
    except Exception as e:                      # 工单四：容错（不暴露堆栈）
        logger.exception("[api_v4] ask 失败")
        raise HTTPException(status_code=500, detail=f"问答失败: {e}")


# ================= 工单四：图像列表 =================
@app.get("/api/v4/images/{doc_id}", response_model=ImagesV4Response,
         tags=["工单四"])
async def get_images(doc_id: str, limit: int = 200):
    """工单四：返回某册已入库图像（path/caption/ocr/vqa，供界面展示）"""
    try:
        from src.image_parser.image_store import ImageStore
        store = ImageStore()
        if not store.client.has_collection(store.collection):
            return ImagesV4Response(doc_id=doc_id, total=0, images=[])
        rows = store.client.query(
            collection_name=store.collection,
            filter=f'doc_id == "{doc_id}"', limit=limit,
            output_fields=["image_id", "page", "path", "caption",
                           "ocr_text", "vqa_text"])
        rows.sort(key=lambda r: (r.get("page", 0), r.get("image_id", "")))
        items = [ImageItem(**{k: r.get(k) or "" for k in
                              ("image_id", "page", "path", "caption",
                               "ocr_text", "vqa_text")}) for r in rows]
        return ImagesV4Response(doc_id=doc_id, total=len(items), images=items)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"图像查询失败: {e}")


# ================= 工单四：上传 PDF（图像管线） =================
@app.post("/api/v4/upload", response_model=UploadV4Response, tags=["工单四"])
async def upload_pdf(pdf: UploadFile = File(...)):
    """工单四：上传 PDF → 图像提取+解析+入库（文本/表格沿用 v3 上传链路）"""
    doc_name = Path(pdf.filename).stem
    save_path = Path("附件") / pdf.filename
    save_path.parent.mkdir(exist_ok=True)
    save_path.write_bytes(await pdf.read())

    # 工单四：复用 Step2/3 脚本化管线（子进程隔离，失败不影响服务）
    manifest = Path("data/images") / f"{doc_name}_images.json"
    extracted = parsed = ingested = 0
    try:
        r = subprocess.run(
            [sys.executable, "-m", "src.image_parser.image_extractor",
             "--pdf", str(save_path), "--out", str(manifest)],
            capture_output=True, text=True, timeout=600, check=True)
        meta = json.loads(manifest.read_text(encoding="utf-8"))
        extracted = len(meta.get("images", []))
    except Exception as e:
        raise HTTPException(500, f"图像提取失败: {e}")

    try:                                        # 工单四：多模态解析（VLM 就绪才有 caption/VQA）
        parsed_path = Path(f"data/image_descriptions/{doc_name}_images_parsed.json")
        subprocess.run(
            [sys.executable, "-m", "src.image_parser.image_parser",
             "--images", str(manifest), "--out", str(parsed_path)],
            capture_output=True, text=True, timeout=3600)
        if parsed_path.exists():
            data = json.loads(parsed_path.read_text(encoding="utf-8"))
            parsed = data.get("stats", {}).get("total", 0)
    except Exception as e:
        logger.warning(f"[api_v4] 图像解析跳过: {e}")

    try:                                        # 工单四：入库 rag_images
        r = subprocess.run(
            [sys.executable, "scripts/init_milvus_v4.py", "--ingest",
             "--rebuild", doc_name],
            capture_output=True, text=True, timeout=1800)
        ingested = extracted if r.returncode == 0 else 0
    except Exception as e:
        logger.warning(f"[api_v4] 入库失败: {e}")

    return UploadV4Response(doc_name=doc_name, images_extracted=extracted,
                            images_parsed=parsed, images_ingested=ingested,
                            message=f"工单四：{doc_name} 图像管线完成（文本/表格请走 /api/v3/upload）")


# ================= 工单四：批量评估 =================
@app.post("/api/v4/evaluate", response_model=EvaluateV4Response, tags=["工单四"])
async def evaluate(req: EvaluateV4Request):
    """工单四：批量跑题并按 expected_keywords 计算命中率准确率"""
    engine = get_engine()
    items: List[EvalItem] = []
    for q in req.questions:
        try:
            if req.use_rag:
                r = engine.ask(q.question, doc_id=q.doc_id)
            else:
                r = engine.ask_llm(q.question)
            answer = r.get("answer", "")
            hits = [kw for kw in q.expected_keywords if kw and kw in answer]
            items.append(EvalItem(
                id=q.id, question=q.question, qtype=q.qtype, answer=answer,
                hit_keywords=hits,
                hit_ratio=round(len(hits) / len(q.expected_keywords), 3)
                if q.expected_keywords else 0.0,
                correct=(len(hits) == len(q.expected_keywords))
                if q.expected_keywords else False,
                latency_ms=r.get("latency_ms", 0.0),
                route=r.get("route", {})))
        except Exception as e:                  # 工单四：单题失败不阻塞整批
            logger.warning(f"[api_v4] 评估题 {q.id} 失败: {e}")
            items.append(EvalItem(id=q.id, question=q.question, qtype=q.qtype,
                                  answer=f"ERROR: {e}"))
    correct = sum(1 for i in items if i.correct)
    total = len(items) or 1
    return EvaluateV4Response(
        total=len(items), correct=correct,
        accuracy=round(correct / total, 3),
        avg_latency_ms=round(sum(i.latency_ms for i in items) / total, 1),
        items=items)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)
