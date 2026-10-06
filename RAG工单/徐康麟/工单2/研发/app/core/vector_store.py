"""向量库：numpy 精确检索（无外部服务、无索引损坏风险），按嵌入模型分目录存放。

工单：人工智能NLP-RAG-基于PDF文档的问答系统优化
阶段：研发 / 索引层（对应 设计/接口设计.md §2.6）

环境事实 2.2：``chromadb`` 不可用且无法安装，故本工单用 numpy 精确检索
（``faiss`` 亦可用，但 2967~3000 条规模下 numpy 更快更稳）。
**索引目录契约**：``研发/data/index/<embedder-slug>-<dim>/``（例 ``bge-m3-1024/``），
目录内文件：``vectors.npy``、``vectors_meta.jsonl``、``meta.json``、``bm25_index.pkl``。
加载时校验 ``meta.json`` 的 embedder 与 dimension，不一致**拒绝加载**（不静默混用）。
"""

from __future__ import annotations

import json
import threading
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from app.core.config import get_settings
from app.core.errors import IndexNotReadyError, StorageError
from app.core.logging_conf import logger, trace
from app.models.schemas import Chunk


class BaseVectorStore(ABC):
    """向量库抽象接口。"""

    @abstractmethod
    def add(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> int:
        """写入分块与向量，返回写入条数。"""

    @abstractmethod
    def search(self, vector: np.ndarray, top_k: int = 10) -> list[tuple[str, float]]:
        """检索，返回 ``[(chunk_id, 余弦分数)]``。"""

    @abstractmethod
    def count(self) -> int:
        """向量条数。"""

    @abstractmethod
    def save(self) -> Path | None:
        """持久化。"""

    @abstractmethod
    def load(self) -> bool:
        """加载，返回是否成功。"""

    @abstractmethod
    def reset(self) -> None:
        """清空。"""


class NumpyVectorStore(BaseVectorStore):
    """numpy 精确余弦检索（向量已 L2 归一化，点积即余弦）。"""

    def __init__(self, index_dir: Path | str | None = None, dimension: int | None = None) -> None:
        settings = get_settings()
        self._settings = settings
        self._dimension = int(dimension or settings.embedding.dimension)
        self._index_dir = Path(index_dir) if index_dir else settings.paths.data_index / "numpy"
        self._vectors: np.ndarray | None = None
        self._chunk_ids: list[str] = []
        self._meta: list[dict[str, Any]] = []
        self._embedder_name = ""
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    @property
    def index_dir(self) -> Path:
        """索引目录。"""
        return self._index_dir

    @property
    def dimension(self) -> int:
        """向量维度。"""
        return self._dimension

    @trace
    def add(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> int:
        """写入分块与向量（维度不一致立即失败，绝不截断/补零）。"""
        try:
            matrix = np.asarray(vectors, dtype=np.float32)
            if matrix.ndim != 2 or matrix.shape[1] != self._dimension:
                raise StorageError(
                    f"向量维度不匹配：期望 {self._dimension}，实际 {list(matrix.shape)}",
                    detail={"index_dir": str(self._index_dir)},
                )
            if len(chunks) != matrix.shape[0]:
                raise StorageError(f"分块数({len(chunks)})与向量数({matrix.shape[0]})不一致")
            with self._lock:
                self._vectors = matrix
                self._chunk_ids = [chunk.chunk_id for chunk in chunks]
                self._meta = [
                    {
                        "chunk_id": chunk.chunk_id,
                        "page": chunk.page,
                        "type": chunk.type,
                        "section": chunk.section,
                        "table_id": chunk.table_id,
                        "is_boilerplate": bool(chunk.is_boilerplate),
                        "keywords": list(chunk.keywords),
                    }
                    for chunk in chunks
                ]
            logger.info(
                "app.core.vector_store",
                "向量写入完成",
                count=len(chunks),
                dimension=self._dimension,
                index_dir=str(self._index_dir),
            )
            return len(chunks)
        except StorageError:
            logger.exception("app.core.vector_store", "向量写入失败（维度/条数校验不通过）")
            raise
        except Exception as exc:
            logger.exception("app.core.vector_store", "向量写入异常")
            raise StorageError(f"向量写入失败: {exc}") from exc

    @trace
    def search(self, vector: np.ndarray, top_k: int = 10) -> list[tuple[str, float]]:
        """numpy 精确检索（点积 = 余弦，因为入库前已 L2 归一化）。"""
        try:
            with self._lock:
                if self._vectors is None or not self._chunk_ids:
                    logger.warning("app.core.vector_store", "向量库为空，检索返回空结果")
                    return []
                query = np.asarray(vector, dtype=np.float32).ravel()
                if query.shape[0] != self._dimension:
                    raise StorageError(f"查询向量维度不匹配：期望 {self._dimension}，实际 {query.shape[0]}")
                norm = float(np.linalg.norm(query))
                if norm > 0:
                    query = query / norm
                scores = self._vectors @ query
                size = min(int(top_k), scores.shape[0])
                # argpartition 取 top-k（O(n)），再局部排序
                if size < scores.shape[0]:
                    candidate_idx = np.argpartition(-scores, size - 1)[:size]
                else:
                    candidate_idx = np.arange(scores.shape[0])
                ordered = sorted(candidate_idx.tolist(), key=lambda i: float(scores[i]), reverse=True)
                return [(self._chunk_ids[i], float(scores[i])) for i in ordered]
        except StorageError:
            logger.exception("app.core.vector_store", "向量检索失败（维度不匹配）")
            raise
        except Exception as exc:
            logger.exception("app.core.vector_store", "向量检索异常")
            raise StorageError(f"向量检索失败: {exc}") from exc

    def count(self) -> int:
        """当前向量条数。"""
        with self._lock:
            return len(self._chunk_ids)

    @trace
    def score_ids(self, vector: np.ndarray, chunk_ids: Sequence[str]) -> dict[str, float]:
        """对指定 chunk_id 精确计算余弦（用于"子块命中→回填父块余弦"）。

        为什么需要它：子块级召回命中的父块可能不在 chunk 级 top-k 里，
        但其原始余弦是置信度判定（拒答阈值）的必需输入。
        """
        try:
            with self._lock:
                if self._vectors is None or not self._chunk_ids:
                    return {}
                query = np.asarray(vector, dtype=np.float32).ravel()
                if query.shape[0] != self._dimension:
                    raise StorageError(f"查询向量维度不匹配：期望 {self._dimension}，实际 {query.shape[0]}")
                norm = float(np.linalg.norm(query))
                if norm > 0:
                    query = query / norm
                index_of = {chunk_id: index for index, chunk_id in enumerate(self._chunk_ids)}
                scores: dict[str, float] = {}
                for chunk_id in chunk_ids:
                    index = index_of.get(chunk_id)
                    if index is None:
                        continue
                    scores[chunk_id] = float(np.dot(self._vectors[index], query))
                return scores
        except StorageError:
            logger.exception("app.core.vector_store", "指定 chunk 余弦计算失败（维度不匹配）")
            raise
        except Exception as exc:
            logger.exception("app.core.vector_store", "指定 chunk 余弦计算异常")
            raise StorageError(f"指定 chunk 余弦计算失败: {exc}") from exc

    @property
    def ready(self) -> bool:
        """是否已加载向量。"""
        return self._vectors is not None and bool(self._chunk_ids)

    # ------------------------------------------------------------------
    @trace
    def save(self) -> Path | None:
        """保存 ``vectors.npy`` / ``vectors_meta.jsonl`` / ``meta.json``。"""
        try:
            with self._lock:
                if self._vectors is None:
                    logger.warning("app.core.vector_store", "向量为空，跳过保存")
                    return None
                self._index_dir.mkdir(parents=True, exist_ok=True)
                np.save(self._index_dir / "vectors.npy", self._vectors)
                with open(self._index_dir / "vectors_meta.jsonl", "w", encoding="utf-8", newline="\n") as handle:
                    for item in self._meta:
                        handle.write(json.dumps(item, ensure_ascii=False) + "\n")
                meta = {
                    "embedder": self._embedder_name,
                    "dimension": self._dimension,
                    "count": len(self._chunk_ids),
                    "built_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
                }
                (self._index_dir / "meta.json").write_text(
                    json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
                )
            logger.info(
                "app.core.vector_store",
                "向量索引已保存",
                index_dir=str(self._index_dir),
                count=len(self._chunk_ids),
                dimension=self._dimension,
            )
            return self._index_dir
        except Exception as exc:
            logger.exception("app.core.vector_store", "向量索引保存失败", index_dir=str(self._index_dir))
            raise StorageError(f"向量索引保存失败: {exc}") from exc

    @trace
    def load(self) -> bool:
        """加载索引；维度不一致返回 False 并记 ERROR（由上层抛 IndexNotReadyError）。"""
        try:
            vector_path = self._index_dir / "vectors.npy"
            meta_path = self._index_dir / "meta.json"
            if not vector_path.exists():
                logger.warning("app.core.vector_store", "向量索引文件不存在", path=str(vector_path))
                return False
            if meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                stored_dim = int(meta.get("dimension", 0))
                self._embedder_name = str(meta.get("embedder", ""))
                if stored_dim and stored_dim != self._dimension:
                    logger.error(
                        "app.core.vector_store",
                        "索引维度与当前嵌入模型不一致，拒绝加载（避免混用不同维度）",
                        index_dir=str(self._index_dir),
                        index_dimension=stored_dim,
                        expected=self._dimension,
                        index_embedder=self._embedder_name,
                    )
                    return False
            vectors = np.load(vector_path)
            with open(self._index_dir / "vectors_meta.jsonl", "r", encoding="utf-8") as handle:
                meta_rows = [json.loads(line) for line in handle if line.strip()]
            if vectors.shape[0] != len(meta_rows):
                logger.error(
                    "app.core.vector_store",
                    "向量条数与元数据条数不一致，拒绝加载",
                    vectors=int(vectors.shape[0]),
                    meta=len(meta_rows),
                )
                return False
            with self._lock:
                self._vectors = np.asarray(vectors, dtype=np.float32)
                self._dimension = int(self._vectors.shape[1])
                self._meta = meta_rows
                self._chunk_ids = [row["chunk_id"] for row in meta_rows]
            logger.info(
                "app.core.vector_store",
                "向量索引加载完成",
                index_dir=str(self._index_dir),
                count=len(self._chunk_ids),
                dimension=self._dimension,
            )
            return True
        except Exception:
            logger.exception("app.core.vector_store", "向量索引加载异常", index_dir=str(self._index_dir))
            return False

    def reset(self) -> None:
        """清空内存索引。"""
        with self._lock:
            self._vectors = None
            self._chunk_ids = []
            self._meta = []


class VectorStore:
    """向量库门面：把嵌入模型信息与索引目录绑定，并提供健康检查。"""

    def __init__(self, embedder=None, index_dir: Path | str | None = None) -> None:
        from app.core.embedder import get_embedder  # 局部导入避免循环依赖

        self._settings = get_settings()
        self._embedder = embedder or get_embedder()
        resolved_dir = Path(index_dir) if index_dir else self._settings.index_dir(
            self._embedder.slug, self._embedder.dimension
        )
        self._store = NumpyVectorStore(index_dir=resolved_dir, dimension=self._embedder.dimension)
        self._store._embedder_name = self._embedder.name

    @property
    def name(self) -> str:
        """后端名。"""
        return "numpy"

    @property
    def dimension(self) -> int:
        """向量维度。"""
        return self._store.dimension

    @property
    def index_dir(self) -> Path:
        """索引目录。"""
        return self._store.index_dir

    def add(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> int:
        """写入向量。"""
        return self._store.add(chunks, vectors)

    def search(self, vector: np.ndarray, top_k: int = 10) -> list[tuple[str, float]]:
        """检索。"""
        return self._store.search(vector, top_k=top_k)

    def score_ids(self, vector: np.ndarray, chunk_ids: Sequence[str]) -> dict[str, float]:
        """对指定 chunk_id 精确计算余弦。"""
        return self._store.score_ids(vector, chunk_ids)

    def count(self) -> int:
        """条数。"""
        return self._store.count()

    def save(self) -> Path | None:
        """持久化。"""
        return self._store.save()

    def load(self) -> bool:
        """加载。"""
        return self._store.load()

    def reset(self) -> None:
        """清空。"""
        self._store.reset()

    def health(self) -> dict[str, Any]:
        """健康信息：backend/dimension/count/model/index_dir/ready。"""
        return {
            "backend": self.name,
            "dimension": self.dimension,
            "count": self.count(),
            "model": self._embedder.name,
            "index_dir": str(self.index_dir),
            "ready": self._store.ready,
        }

    def require_ready(self) -> None:
        """索引未就绪时抛 ``IndexNotReadyError``（含可执行提示）。"""
        if not self._store.ready:
            raise IndexNotReadyError(
                "向量索引未就绪，请先执行：pwsh -NoProfile -File run_py.ps1 研发/scripts/build_index.py",
                detail={"index_dir": str(self.index_dir), "dimension": self.dimension},
            )


_store: VectorStore | None = None
_store_lock = threading.Lock()


def get_vector_store() -> VectorStore:
    """获取进程级向量库单例。"""
    global _store
    with _store_lock:
        if _store is None:
            _store = VectorStore()
    return _store


def reset_vector_store() -> None:
    """重置单例（测试用）。"""
    global _store
    with _store_lock:
        _store = None
