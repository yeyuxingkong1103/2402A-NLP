# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
# 工单01 - 基于PDF文档的问答系统
"""知识库管理：collections 列表、文档删除、片段抽样预览。"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException

from app.config import settings
from app.core.vectorstore import VectorStore, VectorStoreError
from app.schemas import KBCollectionOut, KBListOut

router = APIRouter(tags=["kb"])


@router.get("/api/kb", response_model=KBListOut)
async def list_kb() -> KBListOut:
    store = VectorStore()
    try:
        names = await asyncio.to_thread(lambda: store.client.list_collections())
        cols = []
        for n in names:
            try:
                rows = await asyncio.to_thread(VectorStore(collection=n).count)
            except Exception:  # noqa: BLE001
                rows = 0
            cols.append(KBCollectionOut(name=n, rows=rows))
        return KBListOut(collections=cols)
    except VectorStoreError as e:
        raise HTTPException(503, f"向量库不可用：{e}") from e
    except Exception as e:  # noqa: BLE001
        raise HTTPException(503, f"向量库不可用：{e}") from e


@router.get("/api/kb/sample")
async def sample_chunks(limit: int = 8, chunk_type: str | None = None) -> dict:
    """抽样预览入库片段，用于人工验收「PDF 解析质量」。"""
    store = VectorStore()
    try:
        expr = f'chunk_type == "{chunk_type}"' if chunk_type else ""
        res = await asyncio.to_thread(
            lambda: store.client.query(
                collection_name=store.collection,
                filter=expr or "id >= 0",
                output_fields=["content", "page_label", "chunk_type", "section_path"],
                limit=limit,
            )
        )
        return {"items": res}
    except Exception as e:  # noqa: BLE001
        raise HTTPException(503, f"查询失败：{e}") from e


@router.get("/api/kb/page/{page_no}")
async def chunks_of_page(page_no: int) -> dict:
    """按页取片段 —— 人工抽查 p21/p22/p30/p128/p152 时用。"""
    store = VectorStore()
    try:
        res = await asyncio.to_thread(
            lambda: store.client.query(
                collection_name=store.collection,
                filter=f"page_no == {int(page_no)}",
                output_fields=["content", "page_label", "chunk_type",
                               "section_path", "chunk_index"],
                limit=50,
            )
        )
        return {"page_no": page_no, "count": len(res), "items": res}
    except Exception as e:  # noqa: BLE001
        raise HTTPException(503, f"查询失败：{e}") from e


@router.delete("/api/kb/doc/{doc_id}")
async def delete_doc(doc_id: str) -> dict:
    store = VectorStore()
    try:
        res = await asyncio.to_thread(
            lambda: store.client.delete(
                collection_name=store.collection, filter=f'doc_id == "{doc_id}"'
            )
        )
        return {"ok": True, "doc_id": doc_id, "result": res}
    except Exception as e:  # noqa: BLE001
        raise HTTPException(503, f"删除失败：{e}") from e
