"""知识库服务：离线解析 → 分块 → 向量化 → Milvus 入库 → MySQL 元数据。

职责：这是 RAG 的"离线侧"（写路径），把原始资料变成可检索的向量与元数据；
在线检索走 retrieval_service / rag.retriever，两者共享同一份 Milvus 集合与 MySQL 元数据。

双写存储：Milvus 存向量+原文（负责相似度召回），MySQL（KnowledgeDoc/KnowledgeChunk）
存文档状态与块映射（负责管理界面展示、可追溯，以及删除时定位并清理向量）。

被谁调用：api/knowledge 路由、初始化脚本（按 KNOWLEDGE_DIRS 建库）。

坑与对策（为什么这么写）：
- 低质量块过滤 + 去重：PDF 解析常产生页眉页脚/空行等噪声块，重复块还会让检索结果高度同质化，
  故入库前先按 token 下限和内容指纹过滤；
- 知识变更后失效检索缓存：否则缓存里的旧片段会继续被召回，"更新知识库"看起来没生效；
- 入库用 try/except + rollback + 状态置 failed：解析/向量化异常时不能让文档卡在 processing。
"""
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import delete as sa_delete
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.core.config import settings
from src.core.exceptions import NotFoundError
from src.core.logging import get_logger
from src.db import milvus as milvus_db
from src.db import redis as redis_db
from src.models import KnowledgeChunk, KnowledgeDoc
from src.rag.chunker import chunk_text, estimate_tokens, summarize_chunk
from src.rag.embedder import get_embedder
from src.rag.parser import parse_file

logger = get_logger("service.knowledge")

MIN_CHUNK_TOKENS = 20       # 低质量块阈值


# ---------------- 文档元数据 ----------------
def create_doc_record(db: Session, persona_id: int, title: str, source: str,
                      file_path: str, file_type: str) -> KnowledgeDoc:
    """先落一条 processing 状态的文档记录再开始解析：即使后续解析崩溃，管理端也能看到失败记录。
    各字段按数据库列长截断，防止超长文本触发 MySQL 写入报错。
    """
    doc = KnowledgeDoc(
        persona_id=persona_id, title=title[:255], source=source[:512],
        file_path=file_path[:512], file_type=file_type, status="processing", chunk_count=0,
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)
    return doc


def update_doc_status(db: Session, doc: KnowledgeDoc, status: str, chunk_count: int = 0,
                      error_msg: Optional[str] = None) -> None:
    """更新入库状态（processing/success/failed）与块数；error_msg 空串统一归一为 None 便于判空。"""
    doc.status = status
    doc.chunk_count = chunk_count
    doc.error_msg = (error_msg or "")[:2000] or None
    db.commit()


def list_docs(db: Session, persona_id: Optional[int] = None) -> List[KnowledgeDoc]:
    """列出知识文档，可只列某角色的；按 id 倒序让管理端"最新导入的排最前"。"""
    stmt = select(KnowledgeDoc).order_by(KnowledgeDoc.id.desc())
    if persona_id:
        stmt = stmt.where(KnowledgeDoc.persona_id == persona_id)
    return list(db.execute(stmt).scalars().all())


def get_doc(db: Session, doc_id: int) -> KnowledgeDoc:
    """按 id 取文档，不存在直接抛 NotFoundError（由全局异常处理器转成 404）。"""
    doc = db.get(KnowledgeDoc, doc_id)
    if not doc:
        raise NotFoundError(f"知识文档不存在：{doc_id}")
    return doc


def delete_doc(db: Session, doc_id: int, remove_vectors: bool = True) -> Dict[str, int]:
    """删除文档：先删 Milvus 向量（按 doc_id+persona_id 定位），再删 MySQL 的块记录与文档记录，
    最后失效该角色的检索缓存——保证"删了就搜不到"，不会残留旧向量继续被召回。"""
    doc = get_doc(db, doc_id)
    vector_deleted = 0
    if remove_vectors:
        vector_deleted = milvus_db.delete_knowledge_by_doc(doc.id, doc.persona_id)
    db.execute(sa_delete(KnowledgeChunk).where(KnowledgeChunk.doc_id == doc.id))
    db.delete(doc)
    db.commit()
    redis_db.clear_retrieval_cache(doc.persona_id)  # 阶段 4：知识变更后失效检索缓存
    logger.info("删除知识文档 doc_id=%s 向量=%d", doc_id, vector_deleted)
    return {"doc_id": doc_id, "vectors_deleted": vector_deleted}


# ---------------- 离线入库 ----------------
def _filter_chunks(chunks: List) -> List:
    """低质量/重复块过滤（K-12）：剔除噪声块，并按"去掉空白后的内容"去重。
    为什么要去空白再比：同一段内容在 PDF 里可能因换行/空格差异而重复出现，
    归一化后才能识别；重复块入库会让检索 Top-K 被同一内容占满，降低多样性。"""
    result, seen = [], set()
    for chunk in chunks:
        content = (chunk.content or "").strip()
        if estimate_tokens(content) < MIN_CHUNK_TOKENS:
            continue
        key = "".join(content.split())
        if key in seen:
            continue
        seen.add(key)
        result.append(chunk)
    return result


def ingest_file(db: Session, file_path: str, persona_id: int,
                strategy: str = "paragraph",
                title: Optional[str] = None) -> Dict[str, Any]:
    """解析并入库单个文档（K-12 动态更新的单文件入口）。

    流程：解析 → 建文档记录 → 分块过滤 → 批量向量化 → 双写 Milvus/MySQL → 置 success。
    返回 {doc_id, status, chunk_count, stored}；文件不存在/内容过少会抛错，由调用方汇总。
    """
    if not os.path.exists(file_path):
        raise NotFoundError(f"文件不存在：{file_path}")

    parsed = parse_file(file_path)
    if parsed.char_count < 50:
        raise ValueError(f"文档内容过少或解析失败（可能是扫描件）：{os.path.basename(file_path)}")

    doc = create_doc_record(
        db, persona_id, title or parsed.title, os.path.basename(file_path),
        file_path, parsed.file_type,
    )
    try:
        chunks = _filter_chunks(chunk_text(parsed.text, strategy=strategy))
        if not chunks:
            update_doc_status(db, doc, "failed", 0, "分块结果为空")
            return {"doc_id": doc.id, "status": "failed", "chunk_count": 0, "stored": 0}

        embedder = get_embedder()
        start = time.time()
        vectors = embedder.encode([c.content for c in chunks])
        logger.info("文档向量化完成 doc=%s 块数=%d 耗时=%.1fs",
                    doc.id, len(chunks), time.time() - start)

        now = int(time.time())
        records = [
            {
                "persona_id": int(persona_id),
                "doc_id": int(doc.id),
                "chunk_id": int(chunk.chunk_index),
                "text": chunk.content[:65000],
                "summary": (chunk.summary or summarize_chunk(chunk.content))[:2000],
                "source": os.path.basename(file_path)[:500],
                "created_at": now,
                "updated_at": now,
                "vector": vector,
            }
            for chunk, vector in zip(chunks, vectors)
        ]

        milvus_ids = milvus_db.insert_chunks(records)
        for idx, chunk in enumerate(chunks):
            db.add(KnowledgeChunk(
                doc_id=doc.id,
                persona_id=persona_id,
                chunk_id=chunk.chunk_index,
                content=chunk.content,
                summary=(chunk.summary or summarize_chunk(chunk.content))[:2000],
                milvus_id=milvus_ids[idx] if idx < len(milvus_ids) else None,
            ))
        update_doc_status(db, doc, "success", len(chunks))
        redis_db.clear_retrieval_cache(int(persona_id))  # 阶段 4：知识变更后失效检索缓存
        logger.info("入库完成 doc=%s 文件=%s 块数=%d", doc.id, os.path.basename(file_path), len(chunks))
        return {
            "doc_id": doc.id, "title": doc.title, "persona_id": persona_id,
            "status": "success", "chunk_count": len(chunks), "stored": len(milvus_ids),
        }
    except Exception as exc:
        db.rollback()
        update_doc_status(db, doc, "failed", 0, str(exc))
        logger.error("入库失败 %s：%s", file_path, exc)
        raise


def ingest_directory(db: Session, directory: str, persona_id: int,
                     strategy: str = "paragraph", recursive: bool = True) -> List[Dict[str, Any]]:
    """批量导入目录下所有受支持文档。"""
    from src.rag.parser import SUPPORTED_TYPES
    results: List[Dict[str, Any]] = []
    files: List[str] = []
    for root, _, filenames in os.walk(directory):
        for fname in sorted(filenames):
            if os.path.splitext(fname)[1].lower().lstrip(".") in SUPPORTED_TYPES:
                files.append(os.path.join(root, fname))
        if not recursive:
            break

    logger.info("目录 %s 待入库文件 %d 个", directory, len(files))
    for path in files:
        try:
            results.append(ingest_file(db, path, persona_id, strategy=strategy))
        except Exception as exc:
            results.append({"file": path, "status": "failed", "error": str(exc)})
    return results


def rebuild_persona_index(db: Session, persona_id: int, dirs: List[str],
                          drop_existing: bool = False, strategy: str = "paragraph") -> Dict[str, Any]:
    """按角色重建知识库索引（K-11 动态更新）。"""
    docs = list_docs(db, persona_id)
    if drop_existing:
        milvus_db.delete_knowledge_by_persona(persona_id)
        for doc in docs:
            db.execute(sa_delete(KnowledgeChunk).where(KnowledgeChunk.doc_id == doc.id))
            db.delete(doc)
        db.commit()
        logger.warning("已清空角色 %s 的知识库索引与元数据", persona_id)

    results = []
    for directory in dirs:
        if not os.path.isabs(directory):
            directory = os.path.join(settings.project_root, directory)
        if not os.path.isdir(directory):
            logger.warning("知识库目录不存在，跳过：%s", directory)
            continue
        results.extend(ingest_directory(db, directory, persona_id, strategy=strategy))

    success = [r for r in results if r.get("status") == "success"]
    return {
        "persona_id": persona_id,
        "documents": len(results),
        "success": len(success),
        "chunks": sum(r.get("chunk_count", 0) for r in success),
        "details": results,
    }


def knowledge_stats(db: Session) -> Dict[str, Any]:
    """全局知识库统计（总文档数、按角色分组、Milvus 内向量总量），供管理端看板使用。"""
    docs = list_docs(db)
    stats: Dict[str, Any] = {"total_docs": len(docs), "by_persona": {}}
    for doc in docs:
        entry = stats["by_persona"].setdefault(
            doc.persona_id, {"docs": 0, "chunks": 0, "failed": 0}
        )
        entry["docs"] += 1
        entry["chunks"] += doc.chunk_count or 0
        if doc.status == "failed":
            entry["failed"] += 1
    stats["milvus_total"] = milvus_db.count_knowledge()
    return stats


def search(db: Session, persona_id: int, query: str, top_k: int = 5) -> Tuple[List[Dict], str]:
    """知识库检索（供管理/调试接口与评测使用）。"""
    from src.services.retrieval_service import search_with_query_rewrite
    return search_with_query_rewrite(persona_id, query, top_k=top_k, rerank_top_n=top_k)