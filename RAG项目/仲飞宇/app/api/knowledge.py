"""知识库接口：文档上传入库、文档列表。

上传链路：解析 -> 清洗 -> 分块 -> 向量化 -> 写入 Milvus -> 失效 BM25 缓存 -> 登记 SQL。
"""
from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile

from ..core.document import (
    CHUNK_OVERLAP_MAX,
    CHUNK_OVERLAP_MIN,
    CHUNK_SIZE_MAX,
    CHUNK_SIZE_MIN,
    STRATEGIES,
    Chunker,
    DocumentParser,
)
from ..core.document.enhance import filter_low_quality
from ..core.document.ingest import build_chunks, store_chunks
from ..core.logging_config import get_logger
from ..schemas import DocumentInfo, KnowledgeListResponse

router = APIRouter(tags=["knowledge"])
log = get_logger("api.knowledge")


# 刻意用同步 def：解析 / 向量化 / 写 Milvus 全是阻塞 IO，async def 会卡住整个
# 事件循环（连并发对话一起卡）。写成 def 后 FastAPI 自动丢进线程池执行。
@router.post("/knowledge/upload")
def upload_knowledge(
    request: Request,
    file: UploadFile = File(...),
    # 默认角色与 schemas.ChatRequest 保持一致（改一处就要改两处：那边是 /chat 的默认角色，
    # 这里是上传时没带 role_id 的落点，两者不一致会出现「传到 A 角色、对话在 B 角色」）
    role_id: str = Query("psychologist"),
    # 边界值与理由见 app/core/document/chunker.py：上界是上下文预算约束，
    # scripts/ingest.py 走同一组常量。
    chunk_size: int = Query(500, ge=CHUNK_SIZE_MIN, le=CHUNK_SIZE_MAX),
    chunk_overlap: int = Query(50, ge=CHUNK_OVERLAP_MIN, le=CHUNK_OVERLAP_MAX),
    strategy: str = Query("paragraph"),
    summary: bool | None = Query(None, description="生成 chunk 摘要；不传则用 .env 的 SUMMARY_ENABLED"),
):
    p = request.app.state.pipeline
    # 不传（None）→ 跟随全局开关 SUMMARY_ENABLED；传了就按传的来（可显式覆盖掉全局设置）
    use_summary = p.settings.summary_enabled if summary is None else summary
    if strategy not in STRATEGIES:
        raise HTTPException(status_code=400, detail=f"未知分块策略: {strategy}（可选 {sorted(STRATEGIES)}）")
    try:
        p.ensure_role(role_id)  # 校验角色存在
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    data = file.file.read()  # 同步路由里用底层文件对象，等价于 await file.read()
    try:
        doc = DocumentParser().parse_bytes(file.filename, data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    embed = p.embedding.embed_texts if strategy == "semantic" else None
    texts = Chunker(chunk_size=chunk_size, overlap=chunk_overlap, strategy=strategy).chunk(doc["text"], embed=embed)
    texts = filter_low_quality(texts, p.settings.min_chunk_chars)
    if not texts:
        raise HTTPException(status_code=400, detail="文档解析后无有效文本（扫描件 OCR 未识别成功，或已在 .env 关闭 OCR_ENABLED）")

    try:
        chunks = build_chunks(
            texts,
            source=doc["source"],
            title=doc["title"],
            embedding=p.embedding,
            summary=use_summary,
            llm=p.llm,
        )
        count = store_chunks(
            chunks,
            role_id=role_id,
            source=doc["source"],
            title=doc["title"],
            milvus=p.milvus,
            sql=p.sql,
        )
    except Exception as exc:  # noqa: BLE001
        # 依赖运行中掉线（Embedding/向量库/关系库）不该以裸 500 逸出：/chat 在同一情形下
        # 返回 503，接口文档也承诺 503。完整 traceback 进日志，客户端拿到明确信号。
        log.exception("入库失败 role=%s source=%s", role_id, doc["source"])
        raise HTTPException(
            status_code=503,
            detail=f"服务暂时不可用（{type(exc).__name__}），请稍后重试",
        ) from exc
    # BM25 索引是**每进程**缓存（见 hybrid_retriever），新文档要让本进程的关键词召回看见
    p.retriever.invalidate(role_id)

    log.info("入库完成 role=%s source=%s chunks=%d", role_id, doc["source"], count)
    return {"message": "ok", "chunk_count": count, "source": doc["source"], "role_id": role_id}


@router.get("/knowledge/list", response_model=KnowledgeListResponse)
def list_knowledge(request: Request, role_id: str | None = None):
    p = request.app.state.pipeline
    try:
        docs = p.sql.list_documents(role_id)
    except Exception as exc:  # noqa: BLE001
        log.exception("查询文档列表失败（关系库掉线？）")
        raise HTTPException(
            status_code=503,
            detail=f"服务暂时不可用（{type(exc).__name__}），请稍后重试",
        ) from exc
    items = [
        DocumentInfo(
            id=d.id, role_id=d.role_id, source=d.source, title=d.title, chunk_count=d.chunk_count
        )
        for d in docs
    ]
    return {"documents": items, "total": len(items)}
