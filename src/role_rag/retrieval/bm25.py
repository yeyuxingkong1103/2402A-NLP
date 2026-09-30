"""BM25 关键词路：jieba 分词 + rank_bm25.BM25Okapi。

索引常驻内存（按作用域分片），并用 Redis 中的 ``kb:version`` 做失效判定：
每次入库后版本号自增，检索时发现版本变化就重建索引。
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Sequence

from ..config import Config, get_config
from ..logging_conf import get_logger
from ..store.milvus_store import MilvusStore, get_milvus
from ..store.redis_store import RedisStore, get_redis

logger = get_logger(__name__)

_STOPWORDS = {
    "的", "了", "和", "是", "在", "就", "都", "而", "及", "与", "或", "一个", "我们", "你们",
    "他们", "这", "那", "有", "也", "不", "等", "对", "中", "为", "以", "并", "上", "下",
    "the", "a", "an", "of", "to", "and", "or", "is", "are", "was", "were", "in", "on", "for",
}
_TOKEN_KEEP_RE = re.compile(r"^[\w\u4e00-\u9fff+.-]+$")


def tokenize(text: str) -> list[str]:
    """中英混合分词：jieba 精确模式 + 停用词/标点过滤 + 小写化。"""

    import jieba

    tokens: list[str] = []
    for token in jieba.lcut(text or ""):
        value = token.strip().lower()
        if not value or value in _STOPWORDS or not _TOKEN_KEEP_RE.match(value):
            continue
        tokens.append(value)
    return tokens


@dataclass(slots=True)
class BM25Hit:
    chunk_id: str
    score: float
    payload: dict[str, Any]


class BM25Index:
    """单个作用域的 BM25 索引。"""

    def __init__(self, scope: str, rows: Sequence[dict[str, Any]], version: int) -> None:
        self.scope = scope
        self.version = version
        self.built_at = time.time()
        self.payloads: list[dict[str, Any]] = []
        corpus: list[list[str]] = []
        for row in rows:
            tokens = tokenize(str(row.get("text", "")))
            if not tokens:
                continue
            self.payloads.append(dict(row))
            corpus.append(tokens)
        self._bm25 = None
        if corpus:
            from rank_bm25 import BM25Okapi

            self._bm25 = BM25Okapi(corpus, k1=1.5, b=0.75)

    def __len__(self) -> int:
        return len(self.payloads)

    def search(self, query: str, limit: int) -> list[BM25Hit]:
        if self._bm25 is None:
            return []
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        order = sorted(range(len(scores)), key=lambda idx: -float(scores[idx]))[:limit]
        hits: list[BM25Hit] = []
        for index in order:
            score = float(scores[index])
            if score <= 0:
                continue
            payload = self.payloads[index]
            hits.append(
                BM25Hit(
                    chunk_id=f"{payload.get('doc_id', '')}#{payload.get('chunk_index', index)}",
                    score=score,
                    payload=payload,
                )
            )
        return hits

    def stats(self) -> dict[str, Any]:
        return {"scope": self.scope, "docs": len(self.payloads), "version": self.version,
                "built_at": round(self.built_at, 1)}


class BM25Cache:
    """按作用域缓存 BM25 索引，跟随 kb_version 自动失效。"""

    def __init__(
        self,
        config: Config | None = None,
        milvus: MilvusStore | None = None,
        redis: RedisStore | None = None,
    ) -> None:
        self.config = config or get_config()
        self.milvus = milvus or get_milvus(self.config)
        self.redis = redis or get_redis(self.config)
        self._indexes: dict[str, BM25Index] = {}
        self._lock = threading.RLock()
        self.refresh_seconds = float(self.config.get("retrieval.bm25_refresh_seconds", 60))

    def _load_rows(self, scope: str) -> list[dict[str, Any]]:
        fields = ["pk", "text", "doc_id", "doc_title", "section", "source", "scope", "chunk_index", "tags"]
        expr = 'scope == "shared"' if scope == "shared" else f'(scope == "{scope}") or (scope == "shared")'
        rows: list[dict[str, Any]] = []
        offset = 0
        page = 2000
        while True:
            batch = self.milvus.query_chunks(expr, output_fields=fields, limit=page, offset=offset)
            if not batch:
                break
            rows.extend(batch)
            if len(batch) < page:
                break
            offset += page
        return rows

    def get(self, scope: str, force: bool = False) -> BM25Index:
        version = self.redis.kb_version()
        with self._lock:
            cached = self._indexes.get(scope)
            if cached is not None and cached.version == version and not force:
                return cached
            rows = self._load_rows(scope)
            index = BM25Index(scope, rows, version)
            self._indexes[scope] = index
            logger.info("重建 BM25 索引：scope=%s，文档块=%d，kb_version=%d", scope, len(index), version)
            return index

    def invalidate(self, scope: str | None = None) -> None:
        with self._lock:
            if scope:
                self._indexes.pop(scope, None)
            else:
                self._indexes.clear()

    def stats(self) -> list[dict[str, Any]]:
        with self._lock:
            return [index.stats() for index in self._indexes.values()]


_cache: BM25Cache | None = None
_cache_lock = threading.Lock()


def get_bm25_cache(config: Config | None = None) -> BM25Cache:
    global _cache
    with _cache_lock:
        if _cache is None:
            _cache = BM25Cache(config)
        return _cache
