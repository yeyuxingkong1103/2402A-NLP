"""知识库动态更新接口：上传 / 删除 / 列出文档。开发阶段不做鉴权。

domain 参数即「角色名」，经 roles 表映射到 collection_name 定位目标 collection。
"""
from __future__ import annotations

import logging
import os
import tempfile

from fastapi import APIRouter, File, HTTPException, UploadFile

import ingest
import milvus_store
import roles

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin/knowledge", tags=["admin"])


def _resolve_collection(domain: str) -> str:
    """domain（角色名）-> collection_name；角色不存在抛 404。"""
    role = roles.get_role(domain)
    if role is None:
        raise HTTPException(status_code=404, detail=f"角色不存在: {domain}")
    return role["collection_name"]


def _collections_for(domain: str | None) -> tuple[list[str], list[str]]:
    """返回 (collections, domains) 两个对齐列表；domain 为 None 时遍历所有角色。"""
    if domain is not None:
        return [_resolve_collection(domain)], [domain]
    rs = roles.list_roles()
    return [r["collection_name"] for r in rs], [r["name"] for r in rs]


@router.post("/upload")
def upload(file: UploadFile = File(...), domain: str = ...):
    """上传 PDF -> 解析 -> 清洗 -> 入库到 domain 对应 collection。"""
    collection = _resolve_collection(domain)
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(file.file.read())
        tmp_path = tmp.name
    try:
        count = ingest.ingest_pdf(
            tmp_path, source=file.filename, domain=domain, collection=collection
        )
    finally:
        os.unlink(tmp_path)
    ingest.load_keyword_index.cache_clear()  # 上传后失效 BM25 缓存，避免检索旧索引
    return {
        "source": file.filename,
        "domain": domain,
        "collection": collection,
        "chunks": count,
    }


@router.delete("/{source}")
def delete(source: str, domain: str | None = None):
    """按 source 删除；domain 缺省时遍历所有角色 collection。"""
    collections, _ = _collections_for(domain)
    for c in collections:
        milvus_store.delete_by_source(c, source)
    ingest.load_keyword_index.cache_clear()  # 删除后失效 BM25 缓存，避免残留已删 chunk
    return {"deleted": source, "collections": collections}


@router.get("/list")
def list_knowledge(domain: str | None = None):
    """列出文档（source + chunk 数 + created_at）；domain 缺省时遍历所有角色。"""
    collections, domains = _collections_for(domain)
    result = []
    for c, d in zip(collections, domains):
        for doc in milvus_store.list_documents(c):
            result.append({
                "source": doc["source"],
                "chunk_count": doc["chunk_count"],
                "created_at": doc["created_at"],
                "domain": d,
                "collection": c,
            })
    return result
