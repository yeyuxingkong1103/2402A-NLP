# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-图像内容解析及检索优化
src/image_parser/image_store.py —— 工单四 rag_images Milvus 存储模块（新增文件）

Schema（见 docs/02_图像检索优化方案.md §3）：
  id(PK自增) doc_id image_id page path caption ocr_text vqa_text
  embedding(bge-m3 1024d 文本通道) metadata(JSON，含 clip_embedding 512d)
  —— 注：Milvus Lite 不支持双向量字段，CLIP 向量存 metadata 由 Python 侧
     余弦检索补偿（图像库仅数百条，毫秒级，见 search_clip）。

连接策略与工单三 table_store 一致：远程优先，失败降级 Milvus Lite（不影响 rag_chunks/rag_tables）。
"""
import json as _json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger
from pymilvus import MilvusClient, CollectionSchema, DataType, FieldSchema

WORK_ORDER = "人工智能NLP-RAG-图像内容解析及检索优化"

COLLECTION = "rag_images"
DEFAULT_LITE_PATH = "data/milvus_v4_images.db"      # 工单四：Lite 降级库（独立于工单三）


class ImageStore:
    """工单四：rag_images collection 的 insert / search / delete_by_doc_id"""

    def __init__(self, collection: str = COLLECTION,
                 lite_path: str = DEFAULT_LITE_PATH, force_lite: bool = False):
        self.collection = collection
        host = os.getenv("MILVUS_HOST", "127.0.0.1")
        port = os.getenv("MILVUS_PORT", "19530")
        self.client: Optional[MilvusClient] = None
        if not force_lite:                          # 工单四：远程 Milvus 优先（单测可强制 Lite）
            try:
                remote = f"http://{host}:{port}"
                c = MilvusClient(uri=remote, timeout=5)
                _ = c.get_server_version()
                self.client = c
                logger.info(f"[image_store] 远程 Milvus: {remote}")
            except Exception as e:                  # 工单四：降级 Milvus Lite
                logger.warning(f"[image_store] 远程不可用({e})，降级 Lite")
        if self.client is None:
            Path(lite_path).parent.mkdir(parents=True, exist_ok=True)
            self.client = MilvusClient(uri=lite_path)
            logger.info(f"[image_store] Milvus Lite: {lite_path}")

    # ------------------------------------------------------------------
    def ensure_collection(self) -> None:
        """工单四：建表（双向量 + HNSW/COSINE 索引），已存在则跳过"""
        if self.client.has_collection(self.collection):
            return
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="image_id", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="page", dtype=DataType.INT32),
            FieldSchema(name="path", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="caption", dtype=DataType.VARCHAR, max_length=4096),
            FieldSchema(name="ocr_text", dtype=DataType.VARCHAR, max_length=8192),
            FieldSchema(name="vqa_text", dtype=DataType.VARCHAR, max_length=8192),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1024),
            FieldSchema(name="metadata", dtype=DataType.JSON),
        ]
        schema = CollectionSchema(
            fields, description="工单四 图像多模态向量（人工智能NLP-RAG-图像内容解析及检索优化）")
        self.client.create_collection(self.collection, schema=schema)
        # 工单四：文本向量通道建 HNSW 索引（COSINE 配归一化向量）
        index_params = self.client.prepare_index_params()
        index_params.add_index(field_name="embedding", index_type="HNSW",
                               metric_type="COSINE",
                               params={"M": 16, "efConstruction": 200})
        self.client.create_index(self.collection, index_params=index_params)
        try:                                        # 工单四：远程建索引后需 load；Lite 自动加载
            self.client.load_collection(self.collection)
        except Exception as e:
            logger.debug(f"[image_store] load_collection 跳过: {e}")
        logger.info(f"[image_store] 已创建 collection: {self.collection}")

    # ------------------------------------------------------------------
    def insert(self, records: List[Dict[str, Any]]) -> int:
        """工单四：批量插入；CLIP 向量转入 metadata（Lite 兼容单向量字段）"""
        rows = []
        for r in records:
            meta = dict(r.get("metadata") or {})
            if r.get("clip_embedding"):             # 工单四：CLIP 向量入 metadata
                meta["clip_embedding"] = r["clip_embedding"]
            rows.append({
                "doc_id": r["doc_id"], "image_id": r["image_id"],
                "page": int(r.get("page", 0)), "path": r["path"],
                "caption": (r.get("caption") or "")[:4000],
                "ocr_text": (r.get("ocr_text") or "")[:8000],
                "vqa_text": (r.get("vqa_text") or "")[:8000],
                "embedding": r["embedding"],
                "metadata": meta,
            })
        res = self.client.insert(self.collection, rows)
        cnt = res.get("insert_count", 0) if isinstance(res, dict) else 0
        if cnt < len(rows):                         # 工单四：插入异常时重试一次
            try:
                res = self.client.insert(self.collection, rows)
                cnt = res.get("insert_count", 0) if isinstance(res, dict) else cnt
            except Exception as e:
                logger.warning(f"[image_store] 插入重试失败: {e}")
        return cnt

    # ------------------------------------------------------------------
    @staticmethod
    def _doc_filter(doc_ids: Optional[List[str]]) -> str:
        """工单四：构造 Milvus doc_id 过滤表达式（双引号字符串字面量）"""
        if not doc_ids:
            return ""
        quoted = ", ".join(f'"{d}"' for d in doc_ids)
        return f"doc_id in [{quoted}]"

    # ------------------------------------------------------------------
    def search_text(self, query_vec: List[float], top_k: int = 5,
                    doc_ids: Optional[List[str]] = None,
                    use_clip: bool = False) -> List[Dict[str, Any]]:
        """工单四：向量检索——use_clip=True 走 Python 侧 CLIP 余弦（metadata 通道）"""
        if use_clip:
            return self.search_clip(query_vec, top_k=top_k, doc_ids=doc_ids)
        flt = self._doc_filter(doc_ids)
        res = self.client.search(
            collection_name=self.collection, data=[query_vec],
            anns_field="embedding", limit=top_k, filter=flt,
            output_fields=["doc_id", "image_id", "page", "path",
                           "caption", "ocr_text", "vqa_text", "metadata"],
            search_params={"metric_type": "COSINE", "params": {"ef": 64}},
        )
        hits = []
        for hit in (res[0] if res else []):
            e = hit.get("entity", {})
            e["score"] = hit.get("distance", 0.0)
            hits.append(e)
        return hits

    # ------------------------------------------------------------------
    def search_clip(self, query_vec: List[float], top_k: int = 5,
                    doc_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """工单四：CLIP 通道——全量拉取后 Python 余弦（图像库数百条，毫秒级）"""
        flt = self._doc_filter(doc_ids)
        # 工单四（人工智能NLP-RAG-图像内容解析及检索优化）：doc_ids 为空
        # （"全部"文档跨库检索）时 filter 为空表达式，Milvus 2.6 要求空表达式
        # 必须搭配 limit，否则报 code=1100 "empty expression should be used
        # with limit"，导致 CLIP 通道被跳过。图像库仅数百行，1000 即全量。
        rows = self.client.query(
            collection_name=self.collection, filter=flt,
            output_fields=["doc_id", "image_id", "page", "path",
                           "caption", "ocr_text", "vqa_text", "metadata"],
            limit=1000,
        )
        import numpy as np
        qv = np.asarray(query_vec, dtype="float32")
        scored = []
        for row in rows:
            cvec = (row.get("metadata") or {}).get("clip_embedding")
            if not cvec:
                continue                            # 工单四：无 CLIP 向量的记录跳过
            sim = float(np.dot(qv, np.asarray(cvec, dtype="float32")))
            row["score"] = sim
            scored.append(row)
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    # ------------------------------------------------------------------
    def delete_by_doc_id(self, doc_id: str) -> int:
        """工单四：按 doc_id 删除（重建单册时用；flush 使删除对查询/统计立即可见）"""
        res = self.client.delete(self.collection, filter=f'doc_id == "{doc_id}"')
        try:
            # 工单四：删除默认异步可见，紧接 insert/count 时 stats 仍含旧行（实测 3→6），
            # 重建场景数据量极小，flush 同步后行数与检索结果一致
            self.client.flush(self.collection)
        except Exception:
            pass
        return res if isinstance(res, int) else 0

    def count(self) -> int:
        """工单四：collection 行数统计"""
        if not self.client.has_collection(self.collection):
            return 0
        return self.client.get_collection_stats(self.collection)["row_count"]
