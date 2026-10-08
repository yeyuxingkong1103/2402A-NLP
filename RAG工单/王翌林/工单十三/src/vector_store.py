# -*- coding: utf-8 -*-
"""
工单：人工智能NLP-RAG-基于PDF文档的问答系统
src/vector_store.py — Milvus 向量库封装（pymilvus 3.x）
"""
import json, os
from typing import Any, Dict, List, Optional
import numpy as np
from loguru import logger
from pymilvus import MilvusClient, CollectionSchema, FieldSchema, DataType
from pymilvus.milvus_client.index import IndexParams

COLLECTION_NAME = "rag_chunks"
EMBEDDING_DIM = 1024
# 工单二：人工智能NLP-RAG-基于PDF文档的问答系统优化 —— Milvus 索引参数调优（Step 6 工单规范值）
# HNSW M=16：每节点 16 条出边，内存与召回平衡（1565 子块规模下的性价比最优）
# efConstruction=200：建图搜索宽度，保证建图质量
# ef（查询侧）=64：默认搜索宽度，兼顾召回与 P95 延迟
# COSINE metric 与 bge-m3 归一化嵌入完美匹配
_INDEX_PARAMS = {"metric_type": "COSINE", "index_type": "HNSW", "params": {"M": 16, "efConstruction": 200}}
SEARCH_PARAMS = {"ef": 64}  # 工单二：查询侧 HNSW 搜索参数（人工智能NLP-RAG-基于PDF文档的问答系统优化）

class VectorStore:
    def __init__(self, host=None, port=None, collection=COLLECTION_NAME, dim=EMBEDDING_DIM,
                 lite_path="./data/milvus_local.db", index_params=None):
        host = host or os.getenv("MILVUS_HOST", "localhost")
        port = port or int(os.getenv("MILVUS_PORT", "19530"))
        self.collection, self.dim = collection, dim
        self.index_params = index_params or _INDEX_PARAMS
        self.client, self.mode = None, ""
        self._connect(host, port, lite_path)
    def _connect(self, host, port, lite_path):
        remote_uri = f"http://{host}:{port}"
        try:
            c = MilvusClient(uri=remote_uri, timeout=5); _ = c.get_server_version()
            self.client, self.mode = c, "remote"
            logger.info(f"已连接远程 Milvus: {remote_uri}, v={c.get_server_version()}")
            return
        except Exception as e:
            logger.warning(f"远程 Milvus 失败 ({remote_uri}): {e}")
        os.makedirs(os.path.dirname(os.path.abspath(lite_path)), exist_ok=True)
        c = MilvusClient(uri=lite_path)
        self.client, self.mode = c, "lite"
        logger.info(f"已连接 Milvus Lite: {lite_path}")
    def ensure_collection(self):
        assert self.client is not None
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
            schema = CollectionSchema(fields, description="RAG 系统 — PDF 分块向量 （工单：人工智能NLP-RAG-基于PDF文档的问答系统）")
            self.client.create_collection(collection_name=self.collection, schema=schema)
        self._ensure_index()
        if self.mode == "remote":
            try: self.client.load_collection(self.collection)
            except Exception: pass
        return self.collection
    def _ensure_index(self):
        assert self.client is not None
        try: indexes = self.client.list_indexes(self.collection); has_index = len(indexes) > 0
        except Exception: has_index = False
        if not has_index:
            ip = IndexParams()
            ip.add_index(field_name="embedding", index_type=self.index_params.get("index_type", "HNSW"),
                         metric_type=self.index_params.get("metric_type", "COSINE"),
                         params=self.index_params.get("params", {}))
            self.client.create_index(collection_name=self.collection, index_params=ip)
    def drop_collection(self):
        if self.client and self.client.has_collection(self.collection):
            self.client.drop_collection(self.collection)
            logger.info(f"已删除 collection: {self.collection}")
    def get_stats(self):
        if not self.client or not self.client.has_collection(self.collection):
            return {"collection": self.collection, "exists": False}
        try: cnt = self.client.num_entities(self.collection)
        except Exception: cnt = None
        return {"collection": self.collection, "exists": True, "mode": self.mode, "num_entities": cnt}
    def insert_chunks(self, chunks: List[Dict], vectors: np.ndarray) -> int:
        assert self.client is not None
        self.ensure_collection()
        if len(chunks) == 0: return 0
        records = []
        for i, ch in enumerate(chunks):
            meta = {"source": ch.get("source",""), "chunk_index": ch.get("chunk_index",0),
                    "global_index": ch.get("global_index",0), "char_count": ch.get("char_count",0)}
            records.append({"doc_id": ch.get("doc_id",""), "chunk_id": ch.get("chunk_id",""),
                            "content": ch.get("text",""), "page": int(ch.get("page",0)),
                            "embedding": vectors[i].tolist(), "metadata": meta})
        inserted = 0
        for start in range(0, len(records), 256):
            batch = records[start:start+256]
            self.client.insert(collection_name=self.collection, data=batch)
            inserted += len(batch)
        logger.info(f"插入完成: {inserted} 条 → {self.collection}"); return inserted
    def search(self, query_vec: np.ndarray, top_k=5, doc_id=None, output_fields=None) -> List[Dict]:
        assert self.client is not None
        if output_fields is None:
            output_fields = ["doc_id", "chunk_id", "content", "page", "metadata"]
        filter_expr = f'doc_id == "{doc_id}"' if doc_id else None
        results = self.client.search(
            collection_name=self.collection, data=[query_vec.tolist()], limit=top_k,
            output_fields=output_fields, filter=filter_expr,
            search_params={"metric_type": "COSINE", "params": dict(SEARCH_PARAMS)})  # 工单二：统一查询参数（人工智能NLP-RAG-基于PDF文档的问答系统优化）
        out = []
        if results and len(results) > 0:
            for hit in results[0]:
                hit.pop("embedding", None); out.append(hit)
        return out
    def delete_by_doc_id(self, doc_id: str) -> int:
        assert self.client is not None
        if not self.client.has_collection(self.collection): return 0
        self.client.delete(collection_name=self.collection, filter=f'doc_id == "{doc_id}"')
        try: self.client.flush(self.collection)
        except Exception: pass
        return 1
    def close(self):
        if self.client:
            try: self.client.close()
            except Exception: pass
            self.client = None

def import_chunks_json(chunks_json_path: str, vs=None):
    from src.embedding import get_embedder
    with open(chunks_json_path, "r", encoding="utf-8") as f: data = json.load(f)
    chunks = data["chunks"]
    if chunks and "vector" in chunks[0]:
        vectors = np.array([np.array(c["vector"], dtype=np.float32) for c in chunks])
        logger.info(f"使用 chunks 内已有向量, shape={vectors.shape}")
    else:
        embedder = get_embedder(); vectors = embedder.encode([c["text"] for c in chunks])
    doc_id = data.get("doc_id", "")
    for c in chunks: c.setdefault("doc_id", doc_id)
    vs = vs or VectorStore()
    vs.insert_chunks(chunks, vectors)
    return vs
