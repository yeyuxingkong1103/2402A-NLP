"""入库编排：加载知识库文件 → 分块 → 向量化 → 写 Milvus → 刷新缓存版本。"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from ..config import Config, get_config
from ..errors import IngestError
from ..logging_conf import get_logger
from ..models.embedder import get_embedder
from ..roles import RoleRegistry
from ..store.milvus_store import get_milvus
from ..store.redis_store import get_redis
from .chunking import Chunk, chunk_document, chunk_summary
from .loaders import LoadedDoc, iter_documents

logger = get_logger(__name__)

ProgressFn = Callable[[str, dict[str, Any]], None]


class IngestPipeline:
    """把 data/kb/<scope>/ 下的文档灌入 Milvus。"""

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or get_config()
        self.kb_root = self.config.kb_dir
        self.embedder = get_embedder(self.config)
        self.milvus = get_milvus(self.config)
        self.redis = get_redis(self.config)
        self.roles = RoleRegistry.from_config(self.config)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ 工具
    def scopes(self, roles: Iterable[str] | None = None) -> list[str]:
        """返回要入库的知识库目录（角色目录 + 公共目录）。"""

        if roles:
            wanted = [role for role in roles]
        else:
            wanted = self.roles.enabled_ids()
        scopes = list(dict.fromkeys(wanted))
        if bool(self.config.get("ingest.include_shared", True)) and "shared" not in scopes:
            scopes.append("shared")
        return scopes

    def _documents(self, scope: str) -> list[LoadedDoc]:
        documents = list(iter_documents(self.kb_root, scope))
        if not documents:
            logger.warning("知识库目录为空：%s", self.kb_root / scope)
        return documents

    # ------------------------------------------------------------------ 主流程
    def ingest_scope(
        self,
        scope: str,
        recreate: bool = False,
        progress: ProgressFn | None = None,
    ) -> dict[str, Any]:
        started = time.time()
        documents = self._documents(scope)
        chunks: list[Chunk] = []
        doc_stats: list[dict[str, Any]] = []

        for doc in documents:
            produced = chunk_document(
                doc,
                chunk_size=int(self.config.get("ingest.chunk_size", 700)),
                chunk_overlap=int(self.config.get("ingest.chunk_overlap", 120)),
                min_chunk_chars=int(self.config.get("ingest.min_chunk_chars", 80)),
                max_chunk_chars=int(self.config.get("ingest.max_chunk_chars", 1400)),
            )
            chunks.extend(produced)
            doc_stats.append(
                {
                    "title": doc.title,
                    "source": doc.source,
                    "chars": doc.char_len,
                    "chunks": len(produced),
                }
            )
            if progress:
                progress("chunked", {"scope": scope, "title": doc.title, "chunks": len(produced)})

        if recreate:
            removed = self.milvus.delete_chunks(f'scope == "{scope}"')
            logger.info("[%s] 重建模式：先删除旧块 %d 条", scope, removed)

        inserted = self._write_chunks(chunks, progress)
        self.milvus.flush()
        report = {
            "scope": scope,
            "documents": len(documents),
            "chunks": len(chunks),
            "inserted": inserted,
            "elapsed": round(time.time() - started, 2),
            "chunk_stats": chunk_summary(chunks),
            "docs": doc_stats,
        }
        logger.info(
            "[%s] 入库完成：%d 篇文档 → %d 块（写入 %d），耗时 %.1fs",
            scope, len(documents), len(chunks), inserted, report["elapsed"],
        )
        return report

    def _write_chunks(self, chunks: Sequence[Chunk], progress: ProgressFn | None = None) -> int:
        if not chunks:
            return 0
        batch_size = int(self.config.get("ingest.batch_size", 16))
        inserted = 0
        for start in range(0, len(chunks), batch_size):
            batch = list(chunks[start : start + batch_size])
            texts = [chunk.text for chunk in batch]
            dense, sparse = self.embedder.encode(texts)
            rows = []
            for position, chunk in enumerate(batch):
                rows.append(
                    {
                        "dense": dense[position].tolist(),
                        "sparse": sparse[position],
                        "text": chunk.text,
                        "role": chunk.scope,
                        "scope": chunk.scope,
                        "doc_id": chunk.doc_id,
                        "doc_title": chunk.doc_title,
                        "section": chunk.section,
                        "chunk_index": chunk.chunk_index,
                        "source": chunk.source,
                        "tags": chunk.tags,
                        "char_len": chunk.char_len,
                    }
                )
            inserted += self.milvus.insert_chunks(rows)
            if progress:
                progress("embedded", {"done": min(start + batch_size, len(chunks)), "total": len(chunks)})
        return inserted

    def ingest_all(
        self,
        roles: Iterable[str] | None = None,
        recreate: bool = False,
        progress: ProgressFn | None = None,
    ) -> dict[str, Any]:
        """按作用域逐个入库（角色目录 + 公共目录）。"""

        with self._lock:
            self.milvus.ensure_collections(recreate=False)
            started = time.time()
            reports = []
            for scope in self.scopes(roles):
                source_dir = self.kb_root / scope
                if not source_dir.is_dir():
                    logger.warning("跳过不存在的知识库目录：%s", source_dir)
                    continue
                reports.append(self.ingest_scope(scope, recreate=recreate, progress=progress))
            version = self.redis.bump_kb_version()
            cleared = self.redis.clear_cache()
            total_chunks = sum(item["chunks"] for item in reports)
            total_inserted = sum(item["inserted"] for item in reports)
            summary = {
                "scopes": reports,
                "documents": sum(item["documents"] for item in reports),
                "chunks": total_chunks,
                "inserted": total_inserted,
                "elapsed": round(time.time() - started, 2),
                "kb_version": version,
                "cache_cleared": cleared,
                "kb_rows": self._safe_count(),
            }
            self.redis.incr_stat("ingest_runs")
            self.redis.incr_stat("ingest_chunks", total_inserted)
            logger.info(
                "全部入库完成：%d 块（写入 %d），知识库共 %s 条，kb_version=%d",
                total_chunks, total_inserted, summary["kb_rows"], version,
            )
            return summary

    def ingest_files(
        self,
        paths: Sequence[Path],
        role: str,
        scope: str | None = None,
        title: str | None = None,
    ) -> dict[str, Any]:
        """入库零散文件（供 API 上传 / 命令行指定文件使用）。"""

        from .loaders import load_document

        scope = scope or role
        self.milvus.ensure_collections(recreate=False)
        chunks: list[Chunk] = []
        for path in paths:
            doc = load_document(Path(path), role=role, scope=scope)
            if title:
                doc.title = title
            chunks.extend(
                chunk_document(
                    doc,
                    chunk_size=int(self.config.get("ingest.chunk_size", 700)),
                    chunk_overlap=int(self.config.get("ingest.chunk_overlap", 120)),
                    min_chunk_chars=int(self.config.get("ingest.min_chunk_chars", 80)),
                    max_chunk_chars=int(self.config.get("ingest.max_chunk_chars", 1400)),
                )
            )
        if not chunks:
            raise IngestError("没有可入库的内容")
        inserted = self._write_chunks(chunks)
        self.milvus.flush()
        version = self.redis.bump_kb_version()
        cleared = self.redis.clear_cache()
        self.redis.incr_stat("ingest_chunks", inserted)
        return {
            "scope": scope,
            "documents": len(paths),
            "chunks": len(chunks),
            "inserted": inserted,
            "kb_version": version,
            "cache_cleared": cleared,
        }

    def delete_scope(self, scope: str) -> int:
        """删除某个作用域的全部知识块（按角色或 shared）。"""

        removed = self.milvus.delete_chunks(f'scope == "{scope}"')
        self.milvus.flush()
        self.redis.bump_kb_version()
        self.redis.clear_cache()
        logger.warning("已删除作用域 %s 的知识块 %d 条", scope, removed)
        return removed

    def reset(self) -> None:
        """重建两个集合（清空全部知识块与长期记忆）。"""

        self.milvus.ensure_collections(recreate=True)
        version = self.redis.bump_kb_version()
        self.redis.clear_cache()
        logger.warning("已重建集合，kb_version=%d", version)

    def _safe_count(self) -> int:
        try:
            return self.milvus.count_chunks()
        except Exception as exc:  # pragma: no cover
            logger.warning("统计知识块数量失败：%s", exc)
            return -1

    def report(self) -> dict[str, Any]:
        """知识库现状（供 kb_health 工具与 /api/stats 使用）。"""

        distribution = self.milvus.scope_distribution()
        files: dict[str, int] = {}
        for scope in self.scopes():
            directory = self.kb_root / scope
            files[scope] = len(list(directory.rglob("*"))) if directory.is_dir() else 0
        return {
            "kb_root": str(self.kb_root),
            "scopes": self.scopes(),
            "files": files,
            "chunks_by_scope": distribution,
            "kb_version": self.redis.kb_version(),
            "milvus": self.milvus.stats(),
        }


_pipeline: IngestPipeline | None = None
_pipeline_lock = threading.Lock()


def get_ingest_pipeline(config: Config | None = None) -> IngestPipeline:
    global _pipeline
    with _pipeline_lock:
        if _pipeline is None:
            _pipeline = IngestPipeline(config)
        return _pipeline
