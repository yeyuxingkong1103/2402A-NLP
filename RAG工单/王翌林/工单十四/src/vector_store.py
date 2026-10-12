# -*- coding: utf-8 -*-
"""
工单十四：Milvus 向量库封装（pymilvus 3.x，Lite 模式优先）
"""
import json
import os
from typing import Any, Dict, List
import numpy as np
from loguru import logger
from pymilvus import MilvusClient, CollectionSchema, FieldSchema, DataType
from pymilvus.milvus_client.index import IndexParams

from src.config import MILVUS_HOST, MILVUS_PORT, MILVUS_COLLECTION, MILVUS_LITE_PATH, EMBEDDING_DIM

_INDEX_PARAMS = {"metric_type": "COSINE", "index_type": "HNSW", "params": {"M": 16, "efConstruction": 200}}
SEARCH_PARAMS = {"ef": 64}


class VectorStore:
    def __init__(self, host=None, port=None, collection=None, dim=None, lite_path=None):
        self.collection = collection or MILVUS_COLLECTION
        self.dim = dim or EMBEDDING_DIM
        self.client = None
        self.mode = ""
        self._connect(host or MILVUS_HOST, port or MILVUS_PORT, lite_path or MILVUS_LITE_PATH)

    def _connect(self, host, port, lite_path):
        remote_uri = f"http://{host}:{port}"
        try:
            c = MilvusClient(uri=remote_uri, timeout=5)
            _ = c.get_server_version()
            self.client, self.mode = c, "remote"
            logger.info(f"已连接远程 Milvus: {remote_uri}")
            return
        except Exception as e:
            logger.warning(f"远程 Milvus 失败，使用 Lite: {e}")
        os.makedirs(os.path.dirname(os.path.abspath(lite_path)), exist_ok=True)
        c = MilvusClient(uri=lite_path)
        self.client, self.mode = c, "lite"
        logger.info(f"已连接 Milvus Lite: {lite_path}")

    def ensure_collection(self):
        if not self.client.has_collection(self.collection):
            logger.info(f"创建 collection: {self.collection}, dim={self.dim}")
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=64),
                FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=128),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=8192),
                FieldSchema(name="page", dtype=DataType.INT32),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
                FieldSchema(name="metadata", dtype=DataType.JSON),
            ]
            schema = CollectionSchema(fields, description="工单十四 RAG 分块向量")
            self.client.create_collection(collection_name=self.collection, schema=schema)
        self._ensure_index()
        if self.mode == "remote":
            try:
                self.client.load_collection(self.collection)
            except Exception:
                pass
        return self.collection

    def _ensure_index(self):
        try:
            indexes = self.client.list_indexes(self.collection)
            has_index = len(indexes) > 0
        except Exception:
            has_index = False
        if not has_index:
            ip = IndexParams()
            ip.add_index(field_name="embedding", index_type="HNSW", metric_type="COSINE", params={"M": 16, "efConstruction": 200})
            self.client.create_index(collection_name=self.collection, index_params=ip)

    def drop_collection(self):
        if self.client and self.client.has_collection(self.collection):
            self.client.drop_collection(self.collection)

    def get_stats(self):
        if not self.client or not self.client.has_collection(self.collection):
            return {"collection": self.collection, "exists": False}
        try:
            cnt = self.client.num_entities(self.collection)
        except Exception:
            cnt = None
        return {"collection": self.collection, "exists": True, "mode": self.mode, "num_entities": cnt}

    def insert_chunks(self, chunks, vectors):
        self.ensure_collection()
        if len(chunks) == 0:
            return 0
        records = []
        for i, ch in enumerate(chunks):
            meta = {"source": ch.get("source", ""), "chunk_index": ch.get("chunk_index", 0),
                    "char_count": ch.get("char_count", 0)}
            records.append({"doc_id": ch.get("doc_id", ""), "chunk_id": ch.get("chunk_id", ""),
                            "content": ch.get("text", ""), "page": int(ch.get("page", 0)),
                            "embedding": vectors[i].tolist(), "metadata": meta})
        inserted = 0
        for start in range(0, len(records), 256):
            batch = records[start:start + 256]
            self.client.insert(collection_name=self.collection, data=batch)
            inserted += len(batch)
        logger.info(f"插入完成: {inserted} 条 → {self.collection}")
        return inserted

    def search(self, query_vec, top_k=5, doc_id=None, output_fields=None):
        if output_fields is None:
            output_fields = ["doc_id", "chunk_id", "content", "page", "metadata"]
        filter_expr = f'doc_id == "{doc_id}"' if doc_id else None
        results = self.client.search(
            collection_name=self.collection, data=[query_vec.tolist()], limit=top_k,
            output_fields=output_fields, filter=filter_expr,
            search_params={"metric_type": "COSINE", "params": dict(SEARCH_PARAMS)})
        out = []
        if results and len(results) > 0:
            for hit in results[0]:
                hit.pop("embedding", None)
                out.append(hit)
        return out

    def get_all_chunks(self, limit=5000):
        """拉取全量 chunks 用于 BM25 索引构建"""
        try:
            chunks = self.client.query(collection_name=self.collection,
                                       filter="chunk_id != ''",
                                       output_fields=["chunk_id", "content", "page", "metadata"],
                                       limit=limit)
            if isinstance(chunks, list) and chunks and isinstance(chunks[0], dict):
                return chunks
            # 兼容返回格式异常的情况
            logger.warning(f"query 返回格式异常: {type(chunks)}")
            return []
        except Exception as e:
            logger.warning(f"拉取全量 chunks 失败: {e}")
            return []

    def close(self):
        if self.client:
            try:
                self.client.close()
            except Exception:
                pass
            self.client = None
