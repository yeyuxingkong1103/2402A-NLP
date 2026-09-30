# -*- coding: utf-8 -*-
"""向量检索引擎（Milvus）。

- 后端固定 **Milvus**（pymilvus 连接，host/port 见 ``src.config``，
  默认本机 127.0.0.1:19530）。
- Milvus 不可用时 ``is_available()`` 返回 False，检索返回空（不崩）。
- Embedding 函数可注入；默认加载本地 bge-small-zh-v1.5。

本模块只保留检索类；配套部分已拆分：
- ``milvus_common.py``：字段映射解析、地址安全校验
- ``embedding.py``：向量化函数（远端服务 / 本地模型兜底）
"""
from __future__ import annotations

import concurrent.futures
import json
import logging
import os
import threading
import time
from typing import Optional
from urllib.parse import urlparse

from src import config
from src.core.retrieval.base.retriever_base import BaseRetriever, RetrievalResult
from src.core.retrieval.retrieval_log import log_retrieval_results

from .embedding import EmbeddingFn, default_embedding_fn, remote_embedding_fn
from .milvus_common import (
    MilvusFieldMap,
    _MILVUS_CONNECT_TIMEOUT,
    _decode_json,
    _is_vector_field,
    _lazy_import_pymilvus,
    resolve_milvus_field_map,
    validate_milvus_address,
)

logger = logging.getLogger(__name__)


class VectorRetriever(BaseRetriever):
    """Milvus 向量检索引擎。

    Parameters
    ----------
    collection_name : str | None
        集合名，None 用 config.MILVUS_COLLECTION。
    host / port : str | None
        Milvus 连接地址，None 用 config.MILVUS_HOST/PORT。
    embedding_fn : EmbeddingFn | None
        文本 -> 向量的函数；None 时惰性加载本地 bge 模型。
    """

    name = "vector"

    def __init__(self, collection_name: Optional[str] = None,
                 host: Optional[str] = None, port: Optional[str] = None,
                 embedding_fn: Optional[EmbeddingFn] = None):
        self.collection_name = collection_name or config.MILVUS_COLLECTION
        self.host = host or config.MILVUS_HOST
        self.port = str(port or config.MILVUS_PORT)
        self.embedding_fn = embedding_fn
        self.top_k = config.VECTOR_TOP_K

        self._milvus_collection = None
        self._error: Optional[str] = None
        self._embedding_lock = threading.Lock()
        self._milvus_loaded = False
        self._load()

    # ------------------------------------------------------------------
    # 初始化 / 可用性
    # ------------------------------------------------------------------
    def _load(self) -> None:
        """校验地址并连接 Milvus；失败标记不可用（不崩）。"""
        try:
            self.host, self.port = validate_milvus_address(self.host, self.port)
        except ValueError as exc:
            self._error = str(exc)
            logger.warning("Milvus 地址校验失败: %s", self._error)
            return
        if not self._connect_milvus():
            self._error = self._error or "Milvus 连接失败"

    def _connect_milvus(self) -> bool:
        try:
            imported = _lazy_import_pymilvus()
            if imported is None:
                self._error = "pymilvus 未安装"
                return False
            Collection, CollectionSchema, DataType, FieldSchema, connections, utility = imported

            def _do_connect() -> None:
                connections.connect(
                    alias=config.MILVUS_ALIAS,
                    host=self.host,
                    port=self.port,
                )

            # 连接是唯一的阻塞点，放入线程并限时；超时快速标记不可用，
            # 避免 Milvus 不可达时首查白等一个 TCP/gRPC 默认超时。
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            future = executor.submit(_do_connect)
            try:
                future.result(timeout=_MILVUS_CONNECT_TIMEOUT)
            except concurrent.futures.TimeoutError:
                self._error = (
                    f"Milvus 连接超时（>{_MILVUS_CONNECT_TIMEOUT}s）: "
                    f"{self.host}:{self.port}"
                )
                logger.warning(self._error)
                return False
            finally:
                executor.shutdown(wait=False)

            # 连接已建立，集合检查/加载在调用线程完成（不再阻塞）
            if utility.has_collection(self.collection_name):
                self._milvus_collection = Collection(self.collection_name)
                # 连接时加载一次；后续检索不再重复 load（避免每请求一次集内往返）
                self._milvus_collection.load()
                self._milvus_loaded = True
                return True
            self._error = f"Milvus 集合 {self.collection_name} 不存在（连接成功，等待数据就绪）"
            return False
        except Exception as exc:
            logger.warning("Milvus 连接失败: %s", exc)
            self._error = f"Milvus 连接失败: {exc}"
            return False

    def is_available(self) -> bool:
        """引擎可用：Milvus 已连接（embedding 函数在 retrieve 时懒加载）。"""
        return self._milvus_collection is not None

    def _get_embedding_fn(self) -> EmbeddingFn:
        """取 embedding 函数（懒加载 + 线程安全）。"""
        if self.embedding_fn is not None:
            return self.embedding_fn
        with self._embedding_lock:
            if self.embedding_fn is None:
                self.embedding_fn = default_embedding_fn()
        return self.embedding_fn

    # ------------------------------------------------------------------
    # 索引维护（供 preprocessing 模块调用）
    # ------------------------------------------------------------------
    def add_documents(self, texts: list[str], metadatas: list[dict],
                      ids: Optional[list[str]] = None) -> None:
        """批量写入文档（向量化 + 入库）。"""
        if not texts:
            return
        if self._milvus_collection is None and self._error is not None:
            logger.warning("Milvus 不可用（%s），无法写入。", self._error)
            return
        embeddings = self._get_embedding_fn()(texts)
        doc_ids = ids or [str(i) for i in range(len(texts))]
        self._add_milvus(texts, metadatas, doc_ids, embeddings)

    def _add_milvus(self, texts: list[str], metadatas: list[dict],
                    doc_ids: list[str], embeddings: list[list[float]]) -> None:
        imported = _lazy_import_pymilvus()
        if imported is None:
            return
        Collection, CollectionSchema, DataType, FieldSchema, connections, utility = imported
        id_f, vec_f, text_f, name_f = (
            config.MILVUS_ID_FIELD, config.MILVUS_VECTOR_FIELD,
            config.MILVUS_TEXT_FIELD, config.MILVUS_NAME_FIELD,
        )
        dim = len(embeddings[0]) if embeddings else config.EMBEDDING_DIM

        # 首次写入：创建集合 + 索引（字段名与检索契约一致）
        if self._milvus_collection is None:
            fields = [
                FieldSchema(name=id_f, dtype=DataType.VARCHAR,
                            is_primary=True, max_length=128),
                FieldSchema(name=text_f, dtype=DataType.VARCHAR,
                            max_length=8192),
                FieldSchema(name=name_f, dtype=DataType.VARCHAR,
                            max_length=512),
                FieldSchema(name=vec_f, dtype=DataType.FLOAT_VECTOR,
                            dim=dim),
            ]
            schema = CollectionSchema(fields, description="medical chunks")
            self._milvus_collection = Collection(
                self.collection_name, schema, consistency_level="Strong")
            index_params = {
                "index_type": config.MILVUS_INDEX_TYPE,
                "metric_type": config.MILVUS_METRIC_TYPE,
            }
            if config.MILVUS_INDEX_TYPE.upper() != "AUTOINDEX":
                index_params["params"] = {"nlist": config.MILVUS_NLIST}
            self._milvus_collection.create_index(
                field_name=vec_f, index_params=index_params)
            self._milvus_collection.load()
            self._milvus_loaded = True

        # 集合已存在（含同事库）：按集合真实字段构造行，缺向量字段则跳过不崩 
        schema_fields = {f.name for f in self._milvus_collection.schema.fields}
        if vec_f not in schema_fields:
            logger.warning("集合 %s 缺少向量字段 %s，跳过写入。",
                           self.collection_name, vec_f)
            return
        rows = []
        for did, txt, meta, emb in zip(doc_ids, texts, metadatas, embeddings):
            row = {id_f: did, vec_f: emb, text_f: txt}
            if name_f in schema_fields:
                row[name_f] = str(meta.get("name", did))
            rows.append(row)
        self._milvus_collection.insert(rows)

    # ------------------------------------------------------------------
    # 检索接口
    # ------------------------------------------------------------------
    def retrieve(self, query: str, top_k: Optional[int] = None,
                 **kwargs) -> list[RetrievalResult]:
        started_at = time.perf_counter()
        k = top_k or self.top_k
        q = (query or "").strip()
        retrieval_id = str(kwargs.get("retrieval_id", ""))
        log_details = {
            "collection": self.collection_name,
            "host": self.host,
            "requested_top_k": k,
        }
        if not self.is_available():
            logger.warning("向量检索不可用（%s），返回空结果。", self._error)
            log_retrieval_results(
                engine=self.name,
                query=q,
                results=[],
                elapsed_ms=(time.perf_counter() - started_at) * 1000,
                retrieval_id=retrieval_id,
                status="unavailable",
                error=self._error or "向量检索不可用",
                details=log_details,
            )
            return []

        if not q:
            return []

        try:
            embedding_fn = self._get_embedding_fn()
            query_vec = embedding_fn([q])[0]
            results = self._search_milvus(query_vec, k)
            log_retrieval_results(
                engine=self.name,
                query=q,
                results=results,
                elapsed_ms=(time.perf_counter() - started_at) * 1000,
                retrieval_id=retrieval_id,
                status="success" if results else "empty",
                details=log_details,
            )
            return results
        except Exception as exc:
            logger.warning("向量检索执行失败: %s", exc)
            log_retrieval_results(
                engine=self.name,
                query=q,
                results=[],
                elapsed_ms=(time.perf_counter() - started_at) * 1000,
                retrieval_id=retrieval_id,
                status="error",
                error=f"{type(exc).__name__}: {exc}",
                details=log_details,
            )
            return []

    def _search_milvus(self, query_vec: list[float],
                       k: int) -> list[RetrievalResult]:
        if self._milvus_collection is None:
            return []
        # 仅在未加载时 load 一次；连接/写入时已加载，避免每请求一次集内往返。
        # 用 getattr 防御绕过 __init__ 构造的对象（如测试里 object.__new__）。
        if not getattr(self, "_milvus_loaded", False):
            self._milvus_collection.load()
            self._milvus_loaded = True
        try:
            field_map = resolve_milvus_field_map(
                self._milvus_collection.schema.fields
            )
        except ValueError as exc:
            logger.warning("集合 %s schema 无法映射：%s", self.collection_name, exc)
            return []
        # AUTOINDEX 不接收 nprobe 等搜索参数，按索引类型动态构造
        idx_types = {
            idx.field_name: (idx.params or {}).get("index_type", "")
            for idx in self._milvus_collection.indexes
        }
        search_param: dict = {"metric_type": config.MILVUS_METRIC_TYPE}
        if str(idx_types.get(field_map.vector, "")).upper() != "AUTOINDEX":
            search_param["params"] = {"nprobe": 16}
        results = self._milvus_collection.search(
            data=[query_vec],
            anns_field=field_map.vector,
            param=search_param,
            limit=min(k, 100),
            output_fields=list(field_map.output_fields),
        )
        out: list[RetrievalResult] = []
        for rank, hit in enumerate(results[0], start=1):
            entity = hit.entity
            fields = dict(entity.fields) if hasattr(entity, "fields") else {}
            hit_id = fields.get(field_map.id) or hit.id
            chunk_id = (
                fields.get(field_map.chunk_id) if field_map.chunk_id else None
            ) or hit_id
            title = (
                fields.get(field_map.title) if field_map.title else None
            ) or str(chunk_id)
            text = fields.get(field_map.text) if field_map.text else ""
            metadata = _decode_json(
                fields.get(field_map.metadata) if field_map.metadata else {}
            )
            if not isinstance(metadata, dict):
                metadata = {"raw": metadata}
            section_path = _decode_json(
                fields.get(field_map.section_path)
                if field_map.section_path else []
            )
            content = {
                # 文档块与药品实体不是同一粒度；按 chunk 去重，避免同标题块互相覆盖。
                # 使用稳定的 `chunk:{id}` 标识同一物理证据。
                "fusion_key": f"chunk:{chunk_id}",
                "id": str(hit_id),
                "chunk_id": str(chunk_id),
                "name": str(title),
                "title": str(title),
                "document": str(text or ""),
                "text": str(text or ""),
                "metadata": metadata,
                "source": self.name,
                "milvus_fields": fields,
            }
            optional_values = {
                "parent_id": fields.get(field_map.parent_id)
                if field_map.parent_id else None,
                "document_id": fields.get(field_map.document_id)
                if field_map.document_id else None,
                "chunk_type": fields.get(field_map.chunk_type)
                if field_map.chunk_type else None,
                "source_path": fields.get(field_map.source_path)
                if field_map.source_path else None,
                "cleaned_path": fields.get(field_map.cleaned_path)
                if field_map.cleaned_path else None,
                "section_path": section_path,
            }
            content.update({
                key: value for key, value in optional_values.items()
                if value not in (None, "", [], {})
            })
            out.append(RetrievalResult(
                content=content,
                score=float(hit.score),
                source=self.name,
                rank=rank,
                reason=f"Milvus 相似度 {hit.score:.3f}",
            ))
        return out
