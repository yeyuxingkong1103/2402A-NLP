# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
src/table_parser/table_store.py —— 工单三表格 Milvus 存储封装

职责（见 docs/02_表格检索优化方案.md §七、docs/03 §六）：
  1. 管理 Milvus collection `rag_tables`（与工单二 rag_chunks 分库）
  2. 字段：id、doc_id、table_id、content、table_text、page、embedding、metadata
  3. 提供 insert_tables / search_tables / delete_by_doc_id
  4. 支持按 doc_id 过滤检索（多文档隔离）
"""
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger
from pymilvus import MilvusClient, CollectionSchema, FieldSchema, DataType
from pymilvus.milvus_client.index import IndexParams

# 工单三：collection 名与工单二 rag_chunks 区分
TABLE_COLLECTION_NAME = "rag_tables"
TABLE_EMBEDDING_DIM = 1024

# 工单三：HNSW 索引参数（与工单二保持一致，复用调优经验）
_TABLE_INDEX_PARAMS = {
    "metric_type": "COSINE",
    "index_type": "HNSW",
    "params": {"M": 16, "efConstruction": 200},
}
_TABLE_SEARCH_PARAMS = {"ef": 64}


class TableStore:
    """工单三：rag_tables collection 的 CRUD 封装"""

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        collection: str = TABLE_COLLECTION_NAME,
        dim: int = TABLE_EMBEDDING_DIM,
        lite_path: str = "./data/milvus_local.db",
    ):
        host = host or os.getenv("MILVUS_HOST", "localhost")
        port = port or int(os.getenv("MILVUS_PORT", "19530"))
        self.collection = collection
        self.dim = dim
        self.client = None
        self.mode = ""
        self._connect(host, port, lite_path)

    # ================= 连接 =================
    def _connect(self, host: str, port: int, lite_path: str):
        """工单三：优先远程 Milvus，回退 Milvus Lite（与工单二一致）"""
        remote_uri = f"http://{host}:{port}"
        try:
            c = MilvusClient(uri=remote_uri, timeout=5)
            _ = c.get_server_version()
            self.client = c
            self.mode = "remote"
            logger.info(f"[table_store] 远程 Milvus: {remote_uri} v={c.get_server_version()}")
            return
        except Exception as e:
            logger.warning(f"[table_store] 远程 Milvus 失败 ({remote_uri}): {e}")
        os.makedirs(os.path.dirname(os.path.abspath(lite_path)), exist_ok=True)
        c = MilvusClient(uri=lite_path)
        self.client = c
        self.mode = "lite"
        logger.info(f"[table_store] Milvus Lite: {lite_path}")

    # ================= 建表 =================
    def ensure_collection(self):
        """工单三：创建 rag_tables collection（如不存在）并建索引"""
        assert self.client is not None
        if not self.client.has_collection(self.collection):
            logger.info(
                f"[table_store] 创建 collection: {self.collection}, dim={self.dim}"
            )
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64,
                            is_primary=True, auto_id=True),
                FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=64),
                FieldSchema(name="table_id", dtype=DataType.VARCHAR, max_length=128),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=8192),
                FieldSchema(name="table_text", dtype=DataType.VARCHAR, max_length=8192),
                FieldSchema(name="page", dtype=DataType.INT32),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
                FieldSchema(name="metadata", dtype=DataType.JSON),
            ]
            schema = CollectionSchema(
                fields,
                description="工单三：表格向量库 rag_tables "
                            "（人工智能NLP-RAG-PDF文档的表格解析及检索优化）",
            )
            self.client.create_collection(
                collection_name=self.collection, schema=schema
            )
        self._ensure_index()
        if self.mode == "remote":
            try:
                self.client.load_collection(self.collection)
            except Exception:
                pass
        return self.collection

    def _ensure_index(self):
        """工单三：HNSW + COSINE 索引"""
        assert self.client is not None
        try:
            indexes = self.client.list_indexes(self.collection)
            has_index = len(indexes) > 0
        except Exception:
            has_index = False
        if not has_index:
            ip = IndexParams()
            ip.add_index(
                field_name="embedding",
                index_type=_TABLE_INDEX_PARAMS["index_type"],
                metric_type=_TABLE_INDEX_PARAMS["metric_type"],
                params=_TABLE_INDEX_PARAMS["params"],
            )
            self.client.create_index(
                collection_name=self.collection, index_params=ip
            )

    def drop_collection(self):
        """工单三：删除 collection（重建用）"""
        if self.client and self.client.has_collection(self.collection):
            self.client.drop_collection(self.collection)
            logger.info(f"[table_store] 已删除 collection: {self.collection}")

    def get_stats(self) -> Dict[str, Any]:
        """工单三：返回 collection 统计"""
        if not self.client or not self.client.has_collection(self.collection):
            return {"collection": self.collection, "exists": False}
        try:
            cnt = self.client.num_entities(self.collection)
        except Exception:
            cnt = None
        return {
            "collection": self.collection,
            "exists": True,
            "mode": self.mode,
            "num_entities": cnt,
        }

    # ================= 写入 =================
    def insert_tables(
        self,
        table_chunks: List[Dict[str, Any]],
        vectors: np.ndarray,
    ) -> int:
        """工单三：批量写入表格 chunk + 向量

        Args:
            table_chunks: tables_to_texts 输出，含 table_chunk_id/table_id/doc_id/
                          table_text/page_range/caption/headers 等
            vectors: bge-m3 嵌入 (N, 1024)
        Returns:
            插入条数
        """
        assert self.client is not None
        self.ensure_collection()
        if not table_chunks:
            return 0
        if len(vectors) != len(table_chunks):
            raise ValueError(
                f"向量数 {len(vectors)} != 表格数 {len(table_chunks)}"
            )

        records: List[Dict[str, Any]] = []
        for i, tc in enumerate(table_chunks):
            page_range = tc.get("page_range") or [0, 0]
            page = int(page_range[0]) if page_range else 0
            content = self._build_content(tc)
            meta = {
                "company": tc.get("company", ""),
                "doc_type": "招股说明书",
                "caption": tc.get("caption", ""),
                "headers": tc.get("headers", []),
                "row_count": tc.get("row_count", 0),
                "page_range": page_range,
                "table_chunk_id": tc.get("table_chunk_id", ""),
            }
            records.append({
                "doc_id": str(tc.get("doc_id", "")),
                "table_id": str(tc.get("table_id", "")),
                "content": content[:8000],  # VARCHAR(8192) 截断
                "table_text": (tc.get("table_text") or "")[:8000],
                "page": page,
                "embedding": vectors[i].tolist(),
                "metadata": meta,
            })

        inserted = 0
        for start in range(0, len(records), 256):
            batch = records[start:start + 256]
            self.client.insert(collection_name=self.collection, data=batch)
            inserted += len(batch)
        logger.info(
            f"[table_store] 插入 {inserted} 条表格 → {self.collection}"
        )
        return inserted

    @staticmethod
    def _build_content(tc: Dict[str, Any]) -> str:
        """工单三：构造 content 字段（caption + 首几行原始数据，便于 BM25）"""
        parts = [tc.get("caption", "")]
        for r in (tc.get("rows") or [])[:3]:
            parts.append(" | ".join(str(c) for c in r))
        return "\n".join(p for p in parts if p)

    # ================= 检索 =================
    def search_tables(
        self,
        query_vec: np.ndarray,
        top_k: int = 5,
        doc_id: Optional[str] = None,
        company: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """工单三：向量检索表格

        Args:
            query_vec: bge-m3 查询向量 (1024,)
            top_k: 返回条数
            doc_id: 按文档过滤（多文档隔离）
            company: 按公司过滤
        """
        assert self.client is not None
        output_fields = [
            "doc_id", "table_id", "content", "table_text",
            "page", "metadata",
        ]
        filter_expr = self._build_filter(doc_id, company)
        results = self.client.search(
            collection_name=self.collection,
            data=[query_vec.tolist() if hasattr(query_vec, "tolist") else list(query_vec)],
            limit=top_k,
            output_fields=output_fields,
            filter=filter_expr,
            search_params={
                "metric_type": "COSINE",
                "params": dict(_TABLE_SEARCH_PARAMS),
            },
        )
        out: List[Dict[str, Any]] = []
        if results and len(results) > 0:
            for hit in results[0]:
                # hit 可能含 entity / id / distance / score
                entity = hit.get("entity", hit) if isinstance(hit, dict) else hit
                out.append({
                    "doc_id": entity.get("doc_id", ""),
                    "table_id": entity.get("table_id", ""),
                    "content": entity.get("content", ""),
                    "table_text": entity.get("table_text", ""),
                    "page": entity.get("page", 0),
                    "metadata": entity.get("metadata", {}),
                    "score": hit.get("distance", hit.get("score", 0.0))
                            if isinstance(hit, dict) else 0.0,
                })
        return out

    @staticmethod
    def _build_filter(doc_id: Optional[str], company: Optional[str]) -> Optional[str]:
        """工单三：构造 Milvus filter 表达式"""
        exprs = []
        if doc_id:
            exprs.append(f'doc_id == "{doc_id}"')
        # company 在 metadata JSON 里，Milvus JSON 过滤语法：metadata["company"]
        if company:
            exprs.append(f'metadata["company"] == "{company}"')
        return " and ".join(exprs) if exprs else None

    # ================= 关键词精确匹配 =================
    def search_tables_by_keyword(
        self,
        keywords: List[str],
        doc_id: Optional[str] = None,
        top_k: int = 20,
    ) -> List[Dict[str, Any]]:
        """工单三：关键词精确匹配（BM25 替代）

        用 Milvus LIKE 过滤 content 字段，命中关键词的表格直接召回。
        解决短表（如"发行股数 1,670万股"）向量相似度低、被长表挤掉的问题。

        Args:
            keywords: 关键词列表（如 ["发行股数", "发行后总股本"]）
            doc_id: 按文档过滤
            top_k: 返回条数
        """
        assert self.client is not None
        if not keywords:
            return []
        # 构造 OR LIKE 过滤表达式（搜 table_text，含完整行数据）
        like_exprs = []
        for kw in keywords:
            if kw and len(kw) >= 2:
                like_exprs.append(f'table_text like "%{kw}%"')
        if not like_exprs:
            return []
        filter_parts = " or ".join(like_exprs)
        if doc_id:
            filter_parts = f'(doc_id == "{doc_id}") and ({filter_parts})'
        try:
            results = self.client.query(
                collection_name=self.collection,
                filter=filter_parts,
                output_fields=[
                    "doc_id", "table_id", "content", "table_text",
                    "page", "metadata",
                ],
                limit=top_k,
            )
            out: List[Dict[str, Any]] = []
            for r in (results or []):
                content = r.get("table_text") or r.get("content") or ""
                # 关键词命中数（BM25-like 打分，搜 table_text 含完整行数据）
                kw_hits = sum(1 for kw in keywords if kw in content)
                out.append({
                    "doc_id": r.get("doc_id", ""),
                    "table_id": r.get("table_id", ""),
                    "content": content,
                    "table_text": r.get("table_text", ""),
                    "page": r.get("page", 0),
                    "metadata": r.get("metadata", {}),
                    "score": 0.5 + kw_hits * 0.1,  # 基础分 + 命中加权
                    "kw_hits": kw_hits,
                    "source": "table",
                })
            # 按关键词命中数排序
            out.sort(key=lambda x: x.get("kw_hits", 0), reverse=True)
            return out[:top_k]
        except Exception as e:
            logger.warning(f"[table_store] 关键词查询失败: {e}")
            return []

    # ================= 删除 =================
    def delete_by_doc_id(self, doc_id: str) -> int:
        """工单三：按 doc_id 删除（重新入库前清理）"""
        assert self.client is not None
        if not self.client.has_collection(self.collection):
            return 0
        before = self.get_stats().get("num_entities") or 0
        self.client.delete(
            collection_name=self.collection,
            filter=f'doc_id == "{doc_id}"',
        )
        try:
            self.client.flush(self.collection)
        except Exception:
            pass
        after = self.get_stats().get("num_entities") or 0
        deleted = max(0, before - after)
        logger.info(
            f"[table_store] 删除 doc_id={doc_id}: {deleted} 条（{before}→{after}）"
        )
        return deleted

    def close(self):
        if self.client:
            try:
                self.client.close()
            except Exception:
                pass
            self.client = None


# ================= 便捷函数 =================
def ingest_tables_from_json(
    tables_json_path: str,
    store: Optional[TableStore] = None,
) -> int:
    """工单三：从 data/tables/<doc>_tables.json 读入 + 嵌入 + 入库

    Returns:
        插入条数
    """
    from .table_embedding import load_table_chunks_with_vectors

    p = Path(tables_json_path)
    data = json.loads(p.read_text(encoding="utf-8"))
    doc_id = data.get("doc_name") or data.get("doc_id") or p.stem
    chunks, vecs = load_table_chunks_with_vectors(str(p))
    # 确保 doc_id 写进每个 chunk
    for c in chunks:
        c.setdefault("doc_id", doc_id)
    store = store or TableStore()
    # 先删旧再插新（幂等）
    store.delete_by_doc_id(doc_id)
    return store.insert_tables(chunks, vecs)
