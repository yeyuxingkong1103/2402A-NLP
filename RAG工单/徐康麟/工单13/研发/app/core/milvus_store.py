# -*- coding: utf-8 -*-
"""工单4 Milvus 适配器接口（FAISS 为主用，本模块为**可选**的向量库替换实现）。

工单编号：人工智能NLP-RAG 项目-RAG 性能瓶颈识别与优化

定位与纪律（**不得含糊**）：
    * 提示词要求「向量库用 FAISS 或 Milvus」——工单4 选定 **FAISS 为主用**
      （``研发/app/core/vector_store.py``：``IndexFlatIP`` + ``vectors.npy`` + ``faiss.index``）；
    * 本模块提供**同一检索契约**的 Milvus 适配器：``MilvusStore`` 暴露
      ``connect/ensure_collection/upsert_chunks/search/close/describe``，
      与 ``vector_store.search_vectors`` 的返回结构一致（``[(chunk_id, score)]`` 降序）；
    * 环境事实（T1 实测）：Docker 内 ``milvus-standalone 2.5.1`` 监听 19530、healthz=OK，
      但访问 Docker 需要沙箱提权，且 ``pymilvus`` 未安装 → 工单4 **不把它作为运行依赖**
      （``build_index.py`` 不导入本模块；导入本模块也不会 eager import pymilvus）；
    * 任何「不可用」都返回 ``ok=False`` + ``reason``（见 :func:`describe`），
      **禁止**用 ``ok=True`` 掩盖未就绪能力（与 设计/接口设计.md §3.2 的 clip 口径一致）。

用法（部署到有 Milvus 的环境时）：
    from app.core.milvus_store import MilvusStore, collect_rows_from_index
    rows = collect_rows_from_index("研发/data/index/bge-m3_1024")   # 从 FAISS 索引导出
    store = MilvusStore(uri="http://127.0.0.1:19530")
    store.connect().ensure_collection(dim=1024).upsert_chunks(rows)
    hits = store.search(query_vector, top_k=5)                      # [(chunk_id, score)]
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .errors import RagError, StorageError

WORK_ORDER = "人工智能NLP-RAG 项目-RAG 性能瓶颈识别与优化"

DEFAULT_URI = "http://127.0.0.1:19530"
DEFAULT_COLLECTION = "rag_image_qa_chunks"
METRIC_COSINE = "COSINE"


def pymilvus_version() -> str:
    """``pymilvus`` 版本（元数据存在但 import 失败时返回 ``<版本>(present_but_unimportable:...)``）。"""
    version = "unavailable"
    try:
        import importlib.metadata as md

        version = md.version("pymilvus")
    except Exception:  # noqa: BLE001 —— 版本探测失败不阻断，如实标注
        return "unavailable"
    try:
        import pymilvus  # noqa: F401

        return version
    except Exception as exc:  # noqa: BLE001 —— 半残包必须显式标注
        return f"{version}(present_but_unimportable:{type(exc).__name__})"


def is_available() -> bool:
    """``pymilvus`` 是否**真实可用**（元数据存在不算）。"""
    return "unavailable" not in pymilvus_version() and "unimportable" not in pymilvus_version()


@dataclass(slots=True)
class MilvusRow:
    """一条可写入 Milvus 的向量行（与 chunk 契约对齐）。"""

    chunk_id: str
    vector: Sequence[float]
    file_name: str = ""
    page: int = 0
    modality: str = "text"
    modality_sub: str = ""
    text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"chunk_id": self.chunk_id, "vector": list(self.vector), "file_name": self.file_name,
                "page": int(self.page), "modality": self.modality,
                "modality_sub": self.modality_sub or "", "text": self.text}


def collect_rows_from_index(index_dir: Path | str, *,
                            chunks_path: Path | str | None = None) -> list[MilvusRow]:
    """把 FAISS/numpy 索引（``vectors.npy`` + ``ids.json``）导出为 Milvus 行。

    ``chunks_path`` 缺省为索引同级的 ``../../processed/chunks.jsonl``；读不到就只带 id 与向量
    （不编造页码/模态），并在返回值中如实体现（page=0、modality=""）。
    """
    from .text_utils import read_jsonl

    target = Path(index_dir)
    meta = json.loads((target / "ids.json").read_text(encoding="utf-8"))
    vectors = np.load(target / "vectors.npy").astype(np.float32)
    ids = list(meta.get("ids") or [])
    if vectors.ndim != 2 or len(ids) != int(vectors.shape[0]):
        raise RagError(f"索引自洽性失败：vectors={vectors.shape} ids={len(ids)}", code="RAG-3300",
                       stage="index")
    extra: dict[str, dict[str, Any]] = {}
    if chunks_path is None:
        candidate = target.parents[1] / "processed" / "chunks.jsonl"
        chunks_path = candidate if candidate.is_file() else None
    if chunks_path is not None and Path(chunks_path).is_file():
        for row in read_jsonl(chunks_path):
            cid = str(row.get("chunk_id") or "")
            if cid:
                extra[cid] = row
    rows: list[MilvusRow] = []
    for index, cid in enumerate(ids):
        source = extra.get(cid) or {}
        rows.append(MilvusRow(
            chunk_id=cid, vector=vectors[index].tolist(),
            file_name=str(source.get("file_name") or ""), page=int(source.get("page") or 0),
            modality=str(source.get("type") or source.get("modality") or ""),
            modality_sub=str(source.get("modality_sub") or ""),
            text=str(source.get("content") or ""),
        ))
    return rows


class MilvusStore:
    """Milvus 向量库适配器（惰性连接；不可用时所有方法给出明确原因，不静默降级）。"""

    def __init__(self, uri: str = DEFAULT_URI, *, collection: str = DEFAULT_COLLECTION,
                 dim: int = 1024, metric: str = METRIC_COSINE) -> None:
        self.uri = str(uri)
        self.collection = str(collection)
        self.dim = int(dim)
        self.metric = str(metric)
        self._client: Any = None
        self._connected = False
        self._last_error: str | None = None

    # -- 生命周期 --------------------------------------------------------
    def connect(self, timeout: float = 3.0) -> "MilvusStore":
        """连接 Milvus；``pymilvus`` 不可用或连不上时抛 ``StorageError``（含原因）。"""
        version = pymilvus_version()
        if not is_available():
            self._last_error = f"pymilvus 不可用：{version}"
            raise StorageError(f"{self._last_error}（Milvus 为可选后端，FAISS 主路径不受影响）",
                               code="RAG-7100", stage="storage",
                               detail={"uri": self.uri, "pymilvus": version})
        try:
            from pymilvus import MilvusClient

            self._client = MilvusClient(uri=self.uri, timeout=float(timeout))
            self._connected = True
            return self
        except Exception as exc:  # noqa: BLE001 —— 连接失败必须显式抛出，不静默
            self._last_error = f"{type(exc).__name__}: {exc}"
            raise StorageError(f"Milvus 连接失败：{self._last_error}", code="RAG-7101",
                               stage="storage", detail={"uri": self.uri}) from exc

    def close(self) -> None:
        """关闭连接（幂等）。"""
        if self._client is not None:
            try:
                self._client.close()
            except Exception:  # noqa: BLE001 —— 关闭异常不掩盖主流程结果，仅记录
                self._last_error = "close() 抛异常（已忽略，仅记录）"
            finally:
                self._client = None
                self._connected = False

    # -- 集合与写入 ------------------------------------------------------
    def ensure_collection(self) -> "MilvusStore":
        """创建集合（若不存在）：主键 ``chunk_id``(VARCHAR) + ``vector``(FLOAT_VECTOR,dim)。"""
        self._require_client()
        try:
            if not self._client.has_collection(self.collection):
                self._client.create_collection(collection_name=self.collection,
                                               dimension=self.dim, metric_type=self.metric,
                                               auto_id=False, primary_field_name="chunk_id",
                                               id_type="string", vector_field_name="vector")
            return self
        except Exception as exc:  # noqa: BLE001
            raise StorageError(f"Milvus 建集合失败：{type(exc).__name__}: {exc}", code="RAG-7102",
                               stage="storage", detail={"collection": self.collection}) from exc

    def upsert_chunks(self, rows: Sequence[MilvusRow | dict[str, Any]], *, batch_size: int = 256) -> int:
        """批量写入（``upsert`` 语义：同 ``chunk_id`` 覆盖）。返回写入条数。"""
        self._require_client()
        payload = [row.to_dict() if isinstance(row, MilvusRow) else dict(row) for row in rows]
        total = 0
        try:
            for start in range(0, len(payload), max(int(batch_size), 1)):
                batch = payload[start:start + max(int(batch_size), 1)]
                self._client.upsert(collection_name=self.collection, data=batch)
                total += len(batch)
            return total
        except Exception as exc:  # noqa: BLE001
            raise StorageError(f"Milvus 写入失败（已写入 {total}/{len(payload)}）：{type(exc).__name__}: {exc}",
                               code="RAG-7103", stage="storage",
                               detail={"collection": self.collection}) from exc

    # -- 检索 ------------------------------------------------------------
    def search(self, query: np.ndarray, top_k: int = 5, *,
               modality: str | None = None) -> list[tuple[str, float]]:
        """余弦检索，返回 ``[(chunk_id, score)]`` 降序（与 ``vector_store.search_vectors`` 同契约）。"""
        self._require_client()
        vector = np.asarray(query, dtype=np.float32).reshape(-1)
        if vector.shape[0] != self.dim:
            raise RagError(f"查询向量维度 {vector.shape[0]} 与集合维度 {self.dim} 不一致",
                           code="RAG-3301", stage="index")
        expr = f'modality == "{modality}"' if modality else None
        started = time.perf_counter()
        try:
            results = self._client.search(collection_name=self.collection, data=[vector.tolist()],
                                          limit=max(int(top_k), 1), output_fields=["chunk_id"],
                                          filter=expr, search_params={"metric_type": self.metric})
        except Exception as exc:  # noqa: BLE001
            raise StorageError(f"Milvus 检索失败：{type(exc).__name__}: {exc}", code="RAG-7104",
                               stage="retrieval", detail={"collection": self.collection}) from exc
        hits: list[tuple[str, float]] = []
        for hit in (results[0] if results else []):
            cid = str(hit.get("id") or (hit.get("entity") or {}).get("chunk_id") or "")
            hits.append((cid, float(hit.get("distance") or 0.0)))
        self._last_error = None if hits else f"零命中（耗时 {time.perf_counter() - started:.3f}s）"
        return hits

    # -- 状态 ------------------------------------------------------------
    def describe(self) -> dict[str, Any]:
        """状态报告（供 /health 使用；**不可用即 ok=false + reason**）。"""
        version = pymilvus_version()
        info: dict[str, Any] = {
            "backend": "milvus", "uri": self.uri, "collection": self.collection,
            "dim": self.dim, "metric": self.metric, "pymilvus": version,
            "available": is_available(), "connected": self._connected,
            "ok": bool(self._connected), "last_error": self._last_error,
            "role": "可选适配器（主用向量库为 FAISS：研发/data/index/bge-m3_1024）",
        }
        if not is_available():
            info["reason"] = (f"pymilvus 不可用（{version}）；且本机访问 Milvus 端口 19530 需 Docker/沙箱提权，"
                              f"工单4 不将其作为运行依赖")
        elif not self._connected:
            info["reason"] = "pymilvus 可用但尚未连接（调用 connect() 后再查）"
        return info

    def _require_client(self) -> None:
        if self._client is None:
            raise StorageError("Milvus 未连接：请先 connect()", code="RAG-7100", stage="storage",
                               detail={"uri": self.uri, "pymilvus": pymilvus_version()})


def import_faiss_index_to_milvus(index_dir: Path | str, *, uri: str = DEFAULT_URI,
                                 collection: str = DEFAULT_COLLECTION,
                                 dim: int = 1024, batch_size: int = 256,
                                 chunks_path: Path | str | None = None) -> dict[str, Any]:
    """把 FAISS 索引整体导入 Milvus（部署态可选步骤）。失败抛 ``StorageError``，不静默。"""
    rows = collect_rows_from_index(index_dir, chunks_path=chunks_path)
    store = MilvusStore(uri=uri, collection=collection, dim=dim)
    started = time.perf_counter()
    store.connect().ensure_collection()
    try:
        written = store.upsert_chunks(rows, batch_size=batch_size)
    finally:
        store.close()
    return {"rows": len(rows), "written": written, "collection": collection, "uri": uri,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 2)}


def describe() -> dict[str, Any]:
    """模块级状态（无需实例化即可用于 /health）。"""
    return MilvusStore().describe()


__all__ = ["MilvusRow", "MilvusStore", "collect_rows_from_index", "describe", "is_available",
           "pymilvus_version", "import_faiss_index_to_milvus", "DEFAULT_URI", "DEFAULT_COLLECTION"]
