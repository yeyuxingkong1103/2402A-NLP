# -*- coding: utf-8 -*-
"""工单3 向量索引（numpy 精确检索，faiss 可选）（设计/接口设计.md §3.9 冻结）。

工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化

设计要点：
    * 索引按**嵌入模型分目录**存放（`{index_dir}/{model_slug}/`），**不同维度不得混用**：
      加载时校验 `vectors.shape[1] == expected_dim`，不等抛 `IndexDimensionError`（不静默截断）；
    * 相似度 = L2 归一化后的点积（等价余弦）；返回 `(chunk_id, score)` 降序；
    * `allowed_ids` 用于 `file_names` 硬过滤：**检索前**裁剪，过滤后为空直接返回空列表并 WARN；
    * faiss 为可选加速：真实 import 成功且开关打开才用，否则纯 numpy（两者结果需一致）。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .errors import IndexDimensionError, IndexMissingError, RagError
from .chunker import Chunk

WORK_ORDER = "人工智能NLP-RAG-PDF文档的表格解析及检索优化"

VECTORS_FILE = "vectors.npy"
IDS_FILE = "ids.json"


def _lazy_logger(logger: Any, module: str = "vector_store") -> Any:
    if logger is not None:
        return logger
    from .logging_conf import get_logger

    return get_logger(module)


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


@dataclass(slots=True)
class LoadedVectorIndex:
    """已加载的向量索引（内存态；``vectors`` 为 L2 归一化后的 float32 矩阵）。"""

    vectors: np.ndarray
    ids: list[str]
    model: str
    dim: int
    count: int
    updated_at: str
    index_dir: str
    id_to_row: dict[str, int] = field(default_factory=dict)
    faiss_index: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": self.count, "dim": self.dim, "model": self.model,
            "updated_at": self.updated_at, "index_dir": self.index_dir,
            "faiss": self.faiss_index is not None,
        }


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """按行 L2 归一化（零向量保持为 0，不产生 NaN）。"""
    array = np.asarray(matrix, dtype=np.float32)
    if array.ndim == 1:
        norm = float(np.linalg.norm(array))
        return array / norm if norm > 0 else array
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return array / norms


def try_build_faiss(vectors: np.ndarray, *, enabled: bool = True) -> Any | None:
    """尝试构建 faiss 内积索引；真实 import 失败或未启用时返回 ``None``（显式留痕）。"""
    log = _lazy_logger(None)
    if not enabled:
        log.log_event("vector.faiss_skipped", reason="配置未启用 RAG_VECTOR_STORE__USE_FAISS")
        return None
    try:
        import faiss  # 惰性导入：faiss 为可选加速
    except Exception as exc:  # noqa: BLE001 —— 不可用就走 numpy，必须留痕
        log.log_event("vector.faiss_skipped", level="WARNING", reason=f"import faiss 失败：{type(exc).__name__}: {exc}")
        return None
    matrix = l2_normalize(vectors)
    index = faiss.IndexFlatIP(int(matrix.shape[1]))
    index.add(matrix)
    log.log_event("vector.faiss_built", count=int(matrix.shape[0]), dim=int(matrix.shape[1]))
    return index


def build_vector_index(
    chunks: Sequence[Chunk],
    vectors: np.ndarray,
    *,
    index_dir: Path | str,
    model: str,
    logger: Any = None,
    use_faiss: bool = False,
) -> dict[str, Any]:
    """写 `vectors.npy` + `ids.json`，返回索引元信息（行序与 `ids.json` 一致）。"""
    log = _lazy_logger(logger)
    target = Path(index_dir)
    with log.enter("build_vector_index", {"chunks": len(chunks), "vectors": list(np.shape(vectors)),
                                          "index_dir": str(target), "model": model}) as span:
        started = time.perf_counter()
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2:
            raise IndexDimensionError(f"向量矩阵必须是二维，实际 {matrix.shape}",
                                      detail={"shape": list(matrix.shape)})
        if matrix.shape[0] != len(chunks):
            raise IndexDimensionError(f"向量条数 {matrix.shape[0]} 与块数 {len(chunks)} 不一致",
                                      detail={"vectors": int(matrix.shape[0]), "chunks": len(chunks)})
        if matrix.shape[0] == 0:
            raise IndexDimensionError("向量索引为空（0 条），拒绝写出空索引", detail={"count": 0})
        ids = [c.chunk_id for c in chunks]
        if len(set(ids)) != len(ids):
            raise IndexDimensionError("chunk_id 存在重复，拒绝写出索引")
        normalized = l2_normalize(matrix)
        target.mkdir(parents=True, exist_ok=True)
        vectors_path = target / VECTORS_FILE
        ids_path = target / IDS_FILE
        # 原子写：先写 .tmp 再替换，避免中途失败留下半截索引
        # 注意：np.save 会给不带 .npy 后缀的路径**自动追加 .npy**，故临时名必须以 .npy 结尾
        tmp_vectors = target / "vectors.tmp.npy"
        np.save(tmp_vectors, normalized)
        tmp_vectors.replace(vectors_path)
        meta = {
            "ids": ids, "model": model, "dim": int(normalized.shape[1]),
            "count": int(normalized.shape[0]), "updated_at": _now_iso(),
            "normalized": True, "dtype": "float32",
        }
        tmp_ids = target / (IDS_FILE + ".tmp")
        tmp_ids.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp_ids.replace(ids_path)
        faiss_index = try_build_faiss(normalized, enabled=use_faiss)
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        log.log_event("vector.build", count=int(normalized.shape[0]), dim=int(normalized.shape[1]),
                      model=model, elapsed_ms=elapsed_ms, path=str(vectors_path),
                      faiss=faiss_index is not None)
        result = {**meta, "index_dir": str(target), "vectors_path": str(vectors_path),
                  "ids_path": str(ids_path), "elapsed_ms": elapsed_ms,
                  "faiss": faiss_index is not None}
        span.set_output({k: v for k, v in result.items() if k != "ids"})
        return result


def load_vector_index(index_dir: Path | str, *, expected_dim: int | None = None,
                      logger: Any = None, use_faiss: bool = False) -> LoadedVectorIndex:
    """加载向量索引并做维度校验（不匹配抛 ``IndexDimensionError``）。"""
    log = _lazy_logger(logger)
    target = Path(index_dir)
    with log.enter("load_vector_index", {"index_dir": str(target), "expected_dim": expected_dim}) as span:
        started = time.perf_counter()
        vectors_path = target / VECTORS_FILE
        ids_path = target / IDS_FILE
        if not vectors_path.is_file() or not ids_path.is_file():
            raise IndexMissingError(f"向量索引文件缺失：{vectors_path} / {ids_path}",
                                    detail={"index_dir": str(target)})
        matrix = np.load(vectors_path).astype(np.float32)
        meta = json.loads(ids_path.read_text(encoding="utf-8"))
        ids = list(meta.get("ids") or [])
        if matrix.ndim != 2 or len(ids) != matrix.shape[0]:
            raise IndexDimensionError(
                f"索引自洽性校验失败：vectors={matrix.shape} 与 ids={len(ids)} 不匹配",
                detail={"shape": list(matrix.shape), "ids": len(ids)},
            )
        dim = int(matrix.shape[1])
        if expected_dim is not None and dim != int(expected_dim):
            raise IndexDimensionError(
                f"向量维度不匹配：索引 {dim} 维，期望 {expected_dim} 维（禁止静默截断/填充）",
                detail={"index_dim": dim, "expected_dim": int(expected_dim)},
            )
        index = LoadedVectorIndex(
            vectors=l2_normalize(matrix), ids=ids, model=str(meta.get("model") or ""), dim=dim,
            count=int(matrix.shape[0]), updated_at=str(meta.get("updated_at") or ""),
            index_dir=str(target), id_to_row={cid: i for i, cid in enumerate(ids)},
        )
        index.faiss_index = try_build_faiss(index.vectors, enabled=use_faiss)
        log.log_event("vector.load", count=index.count, dim=index.dim,
                      elapsed_ms=round((time.perf_counter() - started) * 1000, 2), model=index.model)
        span.set_output(index.to_dict())
        return index


def search_vectors(index: LoadedVectorIndex, query: np.ndarray, top_k: int,
                   *, allowed_ids: set[str] | None = None, logger: Any = None) -> list[tuple[str, float]]:
    """余弦检索：返回 ``[(chunk_id, score)]`` 降序；``allowed_ids`` 在检索前裁剪。"""
    log = _lazy_logger(logger)
    with log.enter("search_vectors", {"top_k": top_k, "candidates": index.count,
                                      "allowed": len(allowed_ids) if allowed_ids is not None else None}) as span:
        started = time.perf_counter()
        if index.count == 0:
            span.set_output({"hits": 0, "note": "空索引"})
            return []
        vector = l2_normalize(np.asarray(query, dtype=np.float32).reshape(-1))
        if vector.shape[0] != index.dim:
            raise IndexDimensionError(
                f"查询向量维度 {vector.shape[0]} 与索引维度 {index.dim} 不一致",
                detail={"query_dim": int(vector.shape[0]), "index_dim": index.dim},
            )
        if allowed_ids is None:
            scores = index.vectors @ vector
            rows = np.arange(index.count)
        else:
            rows = np.array([index.id_to_row[cid] for cid in allowed_ids if cid in index.id_to_row],
                            dtype=np.int64)
            if rows.size == 0:
                log.log_event("vector.search_empty_filter", level="WARNING",
                              reason="allowed_ids 与索引无交集", allowed=len(allowed_ids))
                span.set_output({"hits": 0, "filtered": True})
                return []
            scores = index.vectors[rows] @ vector
        k = max(int(top_k), 1)
        if rows.size > k:
            top = np.argpartition(-scores, k - 1)[:k]
            order = top[np.argsort(-scores[top])]
        else:
            order = np.argsort(-scores)
        hits = [(index.ids[int(rows[i])], float(scores[i])) for i in order]
        log.log_event("vector.search", top_k=k, candidates=int(rows.size), hits=len(hits),
                      filtered=allowed_ids is not None,
                      elapsed_ms=round((time.perf_counter() - started) * 1000, 2))
        span.set_output({"hits": len(hits), "top_score": hits[0][1] if hits else None})
        return hits


def index_exists(index_dir: Path | str) -> bool:
    """索引目录是否已就绪（供 UI/检索侧给出友好错误）。"""
    target = Path(index_dir)
    return (target / VECTORS_FILE).is_file() and (target / IDS_FILE).is_file()


def describe_index(index_dir: Path | str, *, logger: Any = None) -> dict[str, Any]:
    """读取索引元信息（不加载向量本体），异常统一转 RagError。"""
    log = _lazy_logger(logger)
    target = Path(index_dir)
    try:
        meta = json.loads((target / IDS_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RagError(f"读取索引元信息失败：{target}（{exc}）", code="RAG-3100",
                       stage="index", detail={"index_dir": str(target)}) from exc
    log.log_event("vector.describe", index_dir=str(target), count=meta.get("count"), dim=meta.get("dim"))
    return {k: v for k, v in meta.items() if k != "ids"}
