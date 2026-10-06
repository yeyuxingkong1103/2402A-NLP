"""向量库管理。

工单要求（5.3）：向量库使用 Chroma。

设计要点：
1. **双后端**：``ChromaVectorStore``（生产首选）与 ``NumpyVectorStore``
   （零依赖持久化后端，保存 ``.npy`` + ``.jsonl``）。
2. **接口一致**：两者都实现 ``add / search / save / load / count``，
   上层检索器无需关心具体后端。
3. **自动降级**：Chroma 不可用（未安装/初始化失败）时回退到 numpy 后端，
   并记录原因。
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from app.core.config import get_settings
from app.core.logging_conf import logger, trace
from app.models.schemas import Chunk

try:  # pragma: no cover
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    HAS_CHROMADB = True
except Exception:  # pragma: no cover
    chromadb = None  # type: ignore
    ChromaSettings = None  # type: ignore
    HAS_CHROMADB = False


class BaseVectorStore(ABC):
    """向量库接口。"""

    name: str = "base"

    @abstractmethod
    def add(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> int:
        """写入分块与向量，返回写入条数。"""

    @abstractmethod
    def search(self, vector: np.ndarray, top_k: int = 10) -> list[tuple[str, float]]:
        """向量检索，返回 ``(chunk_id, 相似度)``。"""

    @abstractmethod
    def count(self) -> int:
        """返回向量总数。"""

    @abstractmethod
    def save(self) -> Path | None:
        """持久化（Chroma 自身持久化时返回其目录）。"""

    def reset(self) -> None:
        """清空索引（重建索引用）。"""


class NumpyVectorStore(BaseVectorStore):
    """numpy 内存向量库：矩阵常驻内存 + 磁盘持久化。

    适合本工单规模（单文档数百~数千 chunk）：检索是纯矩阵乘法，
    速度极快（毫秒级），且不需要任何外部服务。
    """

    name = "numpy"

    def __init__(self, index_dir: Path | str | None = None) -> None:
        settings = get_settings()
        self.index_dir = Path(index_dir) if index_dir else settings.paths.data_index
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.vectors_path = self.index_dir / "vectors.npy"
        self.meta_path = self.index_dir / "vectors_meta.jsonl"
        self.matrix: np.ndarray = np.zeros((0, 0), dtype=np.float32)
        self.chunk_ids: list[str] = []
        self.pages: list[int] = []
        self.types: list[str] = []
        self._chunks: dict[str, Chunk] = {}

    # ------------------------------------------------------------------
    @trace
    def add(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> int:
        if len(chunks) != len(vectors):
            raise ValueError(f"分块数与向量数不匹配: {len(chunks)} vs {len(vectors)}")
        if not chunks:
            return 0
        matrix = np.asarray(vectors, dtype=np.float32)
        if self.matrix.size == 0:
            self.matrix = matrix
        else:
            if matrix.shape[1] != self.matrix.shape[1]:
                raise ValueError(f"向量维度不一致: {matrix.shape[1]} vs {self.matrix.shape[1]}")
            self.matrix = np.vstack([self.matrix, matrix])
        for chunk in chunks:
            self.chunk_ids.append(chunk.chunk_id)
            self.pages.append(chunk.page)
            self.types.append(chunk.type)
            self._chunks[chunk.chunk_id] = chunk
        return len(chunks)

    def search(self, vector: np.ndarray, top_k: int = 10) -> list[tuple[str, float]]:
        if self.matrix.size == 0 or not self.chunk_ids:
            return []
        query = np.asarray(vector, dtype=np.float32).reshape(-1)
        if query.shape[0] != self.matrix.shape[1]:
            raise ValueError(f"查询向量维度不匹配: {query.shape[0]} vs {self.matrix.shape[1]}")
        # 向量已归一化，点积即余弦相似度
        scores = self.matrix @ query
        top_k = min(top_k, len(scores))
        # argpartition 取 top-k 再排序，避免全量排序
        candidate_idx = np.argpartition(-scores, top_k - 1)[:top_k]
        ranked = sorted(candidate_idx, key=lambda i: float(scores[i]), reverse=True)
        return [(self.chunk_ids[i], float(scores[i])) for i in ranked]

    def count(self) -> int:
        return len(self.chunk_ids)

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        return self._chunks.get(chunk_id)

    def page_of(self, chunk_id: str) -> int:
        try:
            return self.pages[self.chunk_ids.index(chunk_id)]
        except ValueError:
            return 0

    # ------------------------------------------------------------------
    @trace
    def save(self) -> Path | None:
        if self.matrix.size == 0:
            logger.warning("app.core.vector_store", "向量库为空，跳过保存")
            return None
        np.save(self.vectors_path, self.matrix)
        with open(self.meta_path, "w", encoding="utf-8", newline="\n") as handle:
            for index, chunk_id in enumerate(self.chunk_ids):
                handle.write(
                    json.dumps(
                        {"chunk_id": chunk_id, "page": self.pages[index], "type": self.types[index]},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        logger.info(
            "app.core.vector_store",
            "向量库已保存",
            vectors=str(self.vectors_path),
            meta=str(self.meta_path),
            count=len(self.chunk_ids),
            dimension=int(self.matrix.shape[1]),
        )
        return self.vectors_path

    @trace
    def load(self) -> bool:
        """从磁盘加载；不存在时返回 False（不抛异常）。"""
        if not self.vectors_path.exists() or not self.meta_path.exists():
            logger.warning("app.core.vector_store", "向量库文件不存在", path=str(self.vectors_path))
            return False
        self.matrix = np.load(self.vectors_path).astype(np.float32)
        self.chunk_ids, self.pages, self.types = [], [], []
        with open(self.meta_path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                self.chunk_ids.append(item["chunk_id"])
                self.pages.append(int(item.get("page", 0)))
                self.types.append(item.get("type", "text"))
        if len(self.chunk_ids) != self.matrix.shape[0]:
            raise ValueError(
                f"向量库元数据与矩阵行数不一致: {len(self.chunk_ids)} vs {self.matrix.shape[0]}"
            )
        logger.info(
            "app.core.vector_store",
            "向量库加载完成",
            count=len(self.chunk_ids),
            dimension=int(self.matrix.shape[1]),
        )
        return True

    def reset(self) -> None:
        self.matrix = np.zeros((0, 0), dtype=np.float32)
        self.chunk_ids, self.pages, self.types = [], [], []
        self._chunks.clear()


class ChromaVectorStore(BaseVectorStore):
    """Chroma 持久化向量库（生产首选）。"""

    name = "chroma"

    def __init__(self, index_dir: Path | str | None = None, collection: str = "prospectus") -> None:
        if not HAS_CHROMADB:
            raise RuntimeError("未安装 chromadb")
        settings = get_settings()
        self.index_dir = Path(index_dir) if index_dir else settings.paths.data_index / "chroma"
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection
        self.client = chromadb.PersistentClient(
            path=str(self.index_dir),
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
        )
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name, metadata={"hnsw:space": "cosine"}
        )
        logger.info("app.core.vector_store", "Chroma 集合就绪", path=str(self.index_dir), collection=collection)

    @trace
    def add(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> int:
        if not chunks:
            return 0
        self.collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            embeddings=[vector.tolist() for vector in np.asarray(vectors, dtype=np.float32)],
            documents=[chunk.content for chunk in chunks],
            metadatas=[
                {
                    "doc_id": chunk.doc_id,
                    "page": int(chunk.page),
                    "type": chunk.type,
                    "section": chunk.section[:200],
                    "table_id": chunk.table_id or "",
                }
                for chunk in chunks
            ],
        )
        return len(chunks)

    def search(self, vector: np.ndarray, top_k: int = 10) -> list[tuple[str, float]]:
        if self.count() == 0:
            return []
        try:
            result = self.collection.query(
                query_embeddings=[np.asarray(vector, dtype=np.float32).tolist()],
                n_results=min(top_k, self.count()),
            )
        except Exception as exc:
            # HNSW 段损坏等底层异常：自愈为“清空重建”，避免整个系统不可用
            logger.warning(
                "app.core.vector_store",
                "Chroma 查询失败，已重置集合（需重建索引）",
                error=f"{type(exc).__name__}: {exc}",
            )
            self.reset()
            return []
        ids = (result.get("ids") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        # cosine 距离 -> 相似度
        return [(chunk_id, 1.0 - float(distance)) for chunk_id, distance in zip(ids, distances)]

    def count(self) -> int:
        try:
            return int(self.collection.count())
        except Exception as exc:  # pragma: no cover
            logger.warning(
                "app.core.vector_store",
                "Chroma 集合不可用（可能索引损坏），按 0 处理并重置",
                error=f"{type(exc).__name__}: {exc}",
            )
            self.reset()
            return 0

    def save(self) -> Path | None:
        logger.info("app.core.vector_store", "Chroma 自动持久化，无需显式保存", path=str(self.index_dir))
        return self.index_dir

    def load(self) -> bool:
        """Chroma 是持久化的：直接把已有集合视为已加载。

        返回 ``count() > 0``，让上层据此判断“是否需要重建向量”。
        """
        count = self.count()
        logger.info(
            "app.core.vector_store",
            "Chroma 持久化集合已就绪",
            path=str(self.index_dir),
            collection=self.collection_name,
            count=count,
        )
        return count > 0

    def reset(self) -> None:
        """删除并重建集合。

        注意：``delete_collection`` 之后必须重新 ``get_or_create_collection``，
        否则旧句柄会指向已删除的集合，后续写入会抛
        ``NotFoundError: Collection ... does not exist``。
        """
        try:
            self.client.delete_collection(self.collection_name)
        except Exception as exc:  # pragma: no cover
            logger.warning("app.core.vector_store", "删除 Chroma 集合失败", error=str(exc))
        # 必须重新获取句柄，刷新内部集合 ID
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name, metadata={"hnsw:space": "cosine"}
        )


class VectorStore:
    """向量库门面：自动选择后端。"""

    def __init__(self, backend: str | None = None, index_dir: Path | str | None = None) -> None:
        self.settings = get_settings()
        self.degraded_reason = ""
        self.impl: BaseVectorStore

        resolved = (backend or self.settings.embedding.backend or "numpy").lower()
        if resolved == "auto":
            resolved = "chroma" if HAS_CHROMADB else "numpy"
            if not HAS_CHROMADB:
                self.degraded_reason = "未安装 chromadb，使用 numpy 后端"

        if resolved == "chroma":
            try:
                self.impl = ChromaVectorStore(index_dir=index_dir)
            except Exception as exc:
                self.degraded_reason = f"Chroma 初始化失败: {type(exc).__name__}: {exc}"
                logger.warning("app.core.vector_store", "Chroma 不可用，降级为 numpy 后端", reason=self.degraded_reason)
                self.impl = NumpyVectorStore(index_dir=index_dir)
        else:
            self.impl = NumpyVectorStore(index_dir=index_dir)
            logger.info("app.core.vector_store", "向量库后端已选定", backend=self.impl.name)

    @property
    def name(self) -> str:
        return self.impl.name

    def add(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> int:
        return self.impl.add(chunks, vectors)

    def search(self, vector: np.ndarray, top_k: int = 10) -> list[tuple[str, float]]:
        return self.impl.search(vector, top_k)

    def count(self) -> int:
        return self.impl.count()

    def save(self) -> Path | None:
        return self.impl.save()

    def load(self) -> bool:
        loader = getattr(self.impl, "load", None)
        return bool(loader()) if callable(loader) else False

    def reset(self) -> None:
        self.impl.reset()

    def health(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "count": self.count(),
            "degraded_reason": self.degraded_reason,
            "chromadb_installed": HAS_CHROMADB,
        }


_store: VectorStore | None = None


def get_vector_store(backend: str | None = None, index_dir: Path | str | None = None) -> VectorStore:
    """工厂函数：获取进程级单例向量库。"""
    global _store
    if _store is None or index_dir is not None:
        _store = VectorStore(backend=backend, index_dir=index_dir)
    return _store


def reset_vector_store() -> None:
    """重置单例（测试用）。"""
    global _store
    _store = None
