# -*- coding: utf-8 -*-
"""server/routes_kb.py —— 知识库管理接口。

在链路中的位置：
    浏览器 → 【本文件】 → Milvus（真实存量）+ data/documents.json（登记表）

四个接口：
    GET    /api/kb/overview    统计与文档列表（扫描 Milvus 的真实点数）
    GET    /api/kb/chunks      某份文档的 chunk 明细
    DELETE /api/kb/document    删除文档（Milvus 向量 + 登记信息 + 原始 PDF 三处一起清）
    （辅助）all_points / data_size 两个内部函数

为什么 chunk 数从 Milvus 现算而不读登记表：
    登记表可能过期（手工删过 Milvus 数据、或构建中断）。
    以向量库实际内容为准，界面上显示的"多少块"才和检索行为一致。
"""
from __future__ import annotations

from collections import Counter
from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse

try:
    from ..pipeline import COLLECTION, PDF_DIR
    from ..retrieval import invalidate_cache
    from ..vector_store import delete_vectors, equal_filter, query_vectors
except ImportError:
    from pipeline import COLLECTION, PDF_DIR
    from retrieval import invalidate_cache
    from vector_store import delete_vectors, equal_filter, query_vectors

from .registry import data_size, load_docs, save_docs
from .security import pdf_name
from .state import build_state

router = APIRouter()

def all_points() -> list[dict[str, Any]]:
    """取出 Milvus 里全部向量记录（用于知识库总览统计）。"""
    return query_vectors(COLLECTION, limit=100000)

@router.get("/api/kb/overview")
def kb_overview():
    """知识库总览：以 Milvus 的真实数据为准统计每个文档的 chunk 数。

    返回：
        {"total_points": 总向量数, "total_docs": 文档数,
         "size_bytes": data 目录字节数, "documents": [{filename, chunks, pages, size, time, vectors}]}

    为什么 chunk 数用 Counter 从 Milvus 现算，而不是读 documents.json：
        登记表可能是过期的（比如手工删过 Milvus 数据、或构建中断）。
        以向量库实际内容为准，界面上显示的"这份文档有多少块"才和检索行为一致。
        pages/size/time 这些只有登记表有的字段，则从 registry 里补，
        查不到就给 None —— 页面显示"-"而不是显示错误数字。
    """
    points = all_points()
    registry = {doc["filename"]: doc for doc in load_docs()}
    docs = []
    for source, count in Counter(row["payload"].get("source") for row in points).items():
        info = registry.get(source, {})
        docs.append({
            "filename": source,
            "chunks": count,
            "pages": info.get("pages"),
            "size": info.get("size"),
            "time": info.get("time"),
            "vectors": info.get("vectors", count),  # 登记表没有就退回用实测 chunk 数
        })
    docs.sort(key=lambda item: item["filename"])  # 稳定排序，避免每次刷新顺序乱跳
    return {"total_points": len(points), "total_docs": len(docs), "size_bytes": data_size(), "documents": docs}

@router.get("/api/kb/chunks")
def kb_chunks(source: str):
    """查看某份文档的全部 chunk 明细。

    参数：
        source: 文档文件名（按 source 字段精确过滤）
    返回：
        {"chunks": [...]}，按 (页码, chunk_id) 排序，便于按原文顺序浏览。

    排序很有必要：
        Milvus 的返回顺序由内部分片决定，不排序的话界面上 chunk 顺序是乱的，
        用户没法按顺序通读、也就无法核对切分是否合理。
    """
    rows = query_vectors(COLLECTION, filter_expression=equal_filter("source", source), limit=10000)
    rows.sort(key=lambda row: (row["payload"].get("page", 0), row["payload"].get("chunk_id", "")))
    return {
        "chunks": [
            {
                "chunk_id": row["payload"].get("chunk_id"),
                "page": row["payload"].get("page"),
                "section": row["payload"].get("section"),
                "text": row["payload"].get("text"),
                # 老数据可能没有 summary 字段，兜底现切前 160 字，保证前端不会显示空白
                "summary": row["payload"].get("summary", str(row["payload"].get("text", ""))[:160]),
                "semantic_type": row["payload"].get("semantic_type", ""),
                "important_kwd": row["payload"].get("important_kwd", []),
                "question_kwd": row["payload"].get("question_kwd", []),
            }
            for row in rows
        ]
    }

@router.delete("/api/kb/document")
def delete_document(source: str):
    """删除一份文档：向量、登记信息、原始 PDF 三处一起清。

    参数：
        source: 文档文件名（查询参数）
    返回：
        {"ok": True, "filename": ...}；被拒时 400/409。

    为什么三处都要删：
        只删向量 -> 登记表还在，列表显示一份查不到内容的文档；
        只删登记 -> 磁盘上的 PDF 越堆越多；
        只删文件 -> 登记表和向量都成了幽灵数据。
        三处同删才不留脏数据。

    校验 filename != source：
        防止传入 "a/../b.pdf" 这类经过规范化的路径绕过校验去删别的文件。
    """
    if build_state["running"]:
        return JSONResponse({"ok": False, "error": "构建进行中，请稍候"}, status_code=409)
    filename = pdf_name(source)
    if not filename or filename != source:
        return JSONResponse({"ok": False, "error": "无效的 PDF 文件名"}, status_code=400)

    delete_vectors(COLLECTION, equal_filter("source", filename))
    save_docs([doc for doc in load_docs() if doc["filename"] != filename])
    pdf_path = PDF_DIR / filename
    if pdf_path.exists():
        pdf_path.unlink()
    # 同上：删除后必须让检索缓存失效，否则被删的内容还能被检索到
    invalidate_cache()
    return {"ok": True, "filename": filename}
