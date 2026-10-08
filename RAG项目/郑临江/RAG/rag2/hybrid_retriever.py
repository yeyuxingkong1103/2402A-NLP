# -*- coding: utf-8 -*-
"""Milvus 混合检索（稠密向量 + 原生 BM25 稀疏向量，服务端 RRF 融合）。

- 稠密路：Milvus 向量检索（HNSW + COSINE）负责语义召回；
- 关键词路：Milvus 原生 BM25（sparse 字段由服务端 BM25 函数从 text 自动生成）；
- 融合：``hybrid_search`` + ``RRFRanker``/``WeightedRanker`` 服务端合并，规避分数量纲差异；
- 重排：可选交叉编码器（bge-reranker-v2-m3）二次精排。

本文件同时保留 ``BM25Index`` / ``tokenize`` / ``weighted_rrf`` 等纯内存实现，
供离线 SQLite 库（``rag2.store``）复用。用法示例见项目根 ``example.py``。
"""

from __future__ import annotations

import logging
import math
import re
import threading
import uuid
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

from .device import resolve_device

logger = logging.getLogger("rag2.hybrid_retriever")

# 文本写入 Milvus 时的最大字符数（与 VARCHAR max_length=8192 匹配，留出余量）
MAX_TEXT_LEN = 8000

# 默认重排模型（BAAI 交叉编码器，与 bge-m3 同族；sentence-transformers 懒加载）
DEFAULT_RERANK_MODEL = r"D:\modelscope\bge-reranker-v2-m3"

# Milvus 原生 BM25 相关默认值：
#   sparse 字段名 / BM25 函数名 / 文本分析器 / 稀疏索引类型
SPARSE_FIELD = "sparse"
BM25_FN_NAME = "bm25_fn"
DEFAULT_ANALYZER = "chinese"          # Milvus 内置中文分析器；可改 "english"/"standard"/自定义 tokenizer
DEFAULT_SPARSE_INDEX = "SPARSE_INVERTED_INDEX"  # 也可用 SPARSE_WAND（更快、召回略降）

# 默认停用词：过滤对检索无区分度的高频词
_DEFAULT_STOPWORDS = frozenset({
    "的", "了", "和", "是", "在", "就", "都", "而", "及", "与", "或",
    "一个", "我们", "你们", "他们", "这", "那", "有", "也", "不", "等",
    "对", "中", "为", "以", "并", "上", "下", "被", "把", "从", "到",
    "the", "a", "an", "of", "to", "and", "or", "is", "are", "was",
    "were", "in", "on", "for", "with", "by", "at",
})

# 保留的 token 形态：中英文单词 / 数字 / 常见符号组合
_TOKEN_KEEP_RE = re.compile(r"^[\w\u4e00-\u9fff+.-]+$")


def tokenize(text: str, stopwords: Iterable[str] = _DEFAULT_STOPWORDS) -> list[str]:
    """中英混合分词：jieba 精确模式 + 停用词/标点过滤 + 小写化。"""
    import jieba  # 中文分词库：把连续的中文句子切成有意义的词（如「农业知识」→ [农业, 知识]），是关键词路 BM25 的基础

    tokens: list[str] = []
    for token in jieba.lcut(text or ""):
        value = token.strip().lower()
        if not value or value in stopwords or not _TOKEN_KEEP_RE.match(value):
            continue
        tokens.append(value)
    return tokens


@dataclass
class Hit:
    """一条召回结果（含融合分数与各路排名，便于调试与前端展示）。"""

    chunk_id: str
    text: str
    score: float
    doc_id: str = ""
    source: str = ""
    page: int = 0
    routes: dict[str, dict[str, float]] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, with_text: bool = True, text_limit: int = 600) -> dict[str, Any]:
        """转换为字典（with_text 控制是否附带文本，text_limit 为截断长度）。"""
        payload: dict[str, Any] = {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "source": self.source,
            "page": self.page,
            "score": round(self.score, 6),
            "routes": self.routes,
        }
        payload.update(self.extra)
        if with_text:
            payload["text"] = self.text if len(self.text) <= text_limit else self.text[:text_limit] + "…"
        return payload

    def __repr__(self) -> str:  # pragma: no cover - 便于调试打印
        return (f"Hit(chunk_id={self.chunk_id!r}, score={self.score:.6f}, "
                f"text={self.text[:40]!r}...)")


@dataclass
class FusedItem:
    """融合中间结果：总分 + 每一路的排名与原始分。"""

    key: str
    score: float
    routes: dict[str, dict[str, float]] = field(default_factory=dict)


def weighted_rrf(
    rankings: dict[str, Sequence[str]],
    weights: dict[str, float] | None = None,
    k: int = 60,
    scores: dict[str, dict[str, float]] | None = None,
) -> list[FusedItem]:
    """加权 RRF 融合：score(d) = Σ_r weight_r / (k + rank_r(d))。

    只看排名不看原始分，天然规避稠密余弦与 BM25 量纲不一致；k 默认 60（与 Milvus 一致）。
    """
    if k < 1:
        raise ValueError("RRF 的 k 必须 >= 1")
    weights = weights or {}
    scores = scores or {}
    fused: dict[str, FusedItem] = {}

    for route, ordered in rankings.items():
        weight = float(weights.get(route, 1.0))
        if weight <= 0:
            continue
        route_scores = scores.get(route, {})
        for position, key in enumerate(ordered, start=1):
            item = fused.get(key)
            if item is None:
                item = FusedItem(key=key, score=0.0)
                fused[key] = item
            item.score += weight / (k + position)
            item.routes[route] = {
                "rank": float(position),
                "raw": round(float(route_scores.get(key, 0.0)), 6),
                "weight": weight,
            }
    return sorted(fused.values(), key=lambda item: (-item.score, item.key))


class BM25Index:
    """轻量级 BM25 关键词索引，面向内存中的文本列表构建（自实现打分）。"""

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        stopwords: Iterable[str] | None = None,
        tokenizer: Callable[[str], list[str]] | None = None,
    ) -> None:
        """初始化 BM25 超参数（k1 词频饱和、b 长度归一化、stopwords/tokenizer 可自定义）。"""
        self.k1 = k1
        self.b = b
        self._stopwords = frozenset(stopwords) if stopwords is not None else _DEFAULT_STOPWORDS
        self._tokenizer = tokenizer or (lambda t: tokenize(t, self._stopwords))
        self.docs: list[list[str]] = []
        self.doc_len: list[int] = []
        self.avg_len: float = 0.0
        self.df: dict[str, int] = {}
        self.idf: dict[str, float] = {}
        self._built = False

    @property
    def built(self) -> bool:
        return self._built

    def __len__(self) -> int:
        return len(self.docs)

    def build(self, texts: Sequence[str]) -> None:
        """对文本列表分词并构建索引。"""
        self.docs = [self._tokenizer(t) for t in texts]
        self.doc_len = [len(d) for d in self.docs]
        n = len(self.docs)
        self.avg_len = sum(self.doc_len) / n if n else 0.0

        df: dict[str, int] = {}
        for doc in self.docs:
            for term in set(doc):
                df[term] = df.get(term, 0) + 1
        self.df = df

        # IDF：越稀有的词越能区分文档；+0.5 平滑避免除零。
        self.idf = {term: math.log(1 + (n - cnt + 0.5) / (cnt + 0.5))
                    for term, cnt in df.items()}
        self._built = True
        logger.info("BM25 索引构建完成：文档数=%d", n)

    def search(self, query: str, top_k: int = 20) -> list[tuple[int, float]]:
        """按 BM25 分数检索，返回按分数降序的 (文档下标, 分数) 列表。"""
        if not self._built:
            raise RuntimeError("BM25 索引尚未构建，请先调用 build()")
        q_terms = self._tokenizer(query)
        if not q_terms:
            return []

        scored: list[tuple[int, float]] = []
        for i, doc in enumerate(self.docs):
            score = self._score(q_terms, doc, i)
            if score > 0:
                scored.append((i, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def _score(self, q_terms: list[str], doc: list[str], doc_idx: int) -> float:
        """计算单个文档的 BM25 分数。"""
        tf = Counter(doc)
        doc_len = self.doc_len[doc_idx]
        total = 0.0
        for term in q_terms:
            idf = self.idf.get(term)
            if idf is None:
                continue
            f = tf.get(term, 0)
            if f == 0:
                continue
            # BM25 单词得分 = IDF × 词频饱和因子（k1 控词频饱和，b 控长度归一化）。
            denom = f + self.k1 * (1 - self.b + self.b * doc_len / self.avg_len)
            total += idf * (f * (self.k1 + 1)) / denom
        return total


def _parse_hits(raw: Any) -> list[Hit]:
    """把 Milvus search / hybrid_search 的原始结果解析成 Hit 列表。"""
    hits: list[Hit] = []
    for batch in raw:
        for item in batch:
            entity = item.get("entity", {})
            hits.append(Hit(
                chunk_id=str(entity.get("chunk_id") or item.get("id", "")),
                text=str(entity.get("text", "")),
                doc_id=str(entity.get("doc_id", "")),
                source=str(entity.get("source", "")),
                page=int(entity.get("page", 0) or 0),
                score=float(item.get("distance", 0.0)),
            ))
    return hits


class HybridRetriever:
    """Milvus（稠密） + BM25（关键词）混合检索器。

    一次实例化后复用连接与索引；入库与检索共用同一对象，调用极简。
    """

    def __init__(
        self,
        uri: str = "http://localhost:19530",
        collection: str = "hybrid_docs",
        dim: int | None = 1024,
        metric_type: str = "COSINE",
        embed_fn: Callable[[list[str]], list[list[float]]] | None = None,
        embed_model: str | None = None,
        device: str = "auto",
        sparse_field: str = SPARSE_FIELD,
        analyzer: str = DEFAULT_ANALYZER,
        sparse_index_type: str = DEFAULT_SPARSE_INDEX,
        rrf_k: int = 60,
        dense_weight: float = 1.0,
        bm25_weight: float = 1.0,
        per_doc_limit: int = 0,
        timeout: int = 30,
        rerank_model: str | None = None,
        rerank_fn: Callable[[str, Sequence[str]], Sequence[float]] | None = None,
        rerank_batch_size: int = 32,
        rerank_max_length: int = 512,
    ) -> None:
        """初始化混合检索器（懒连接：首次入库/检索时才连 Milvus）。

        关键参数：embed_fn/embed_model（向量化）、rerank_model/rerank_fn（重排）、
        sparse_field/analyzer（原生 BM25）、rrf_k/dense_weight/bm25_weight（融合权重）、
        per_doc_limit（单文档片段上限）、device（加载设备）。
        """
        self.uri = uri
        self.collection = collection
        self.dim = int(dim) if dim else None
        self.metric_type = metric_type
        self._embed_fn = embed_fn
        self._embed_model = embed_model
        self.device = device
        self.sparse_field = sparse_field
        self.analyzer = analyzer
        self.sparse_index_type = sparse_index_type
        self.rrf_k = int(rrf_k)
        self.dense_weight = float(dense_weight)
        self.bm25_weight = float(bm25_weight)
        self.per_doc_limit = int(per_doc_limit)
        self.timeout = int(timeout)
        self.rerank_model = rerank_model
        self._rerank_fn = rerank_fn
        self.rerank_batch_size = int(rerank_batch_size)
        self.rerank_max_length = int(rerank_max_length)

        self._client: Any = None
        self._collection_ready = False
        self._lock = threading.RLock()
        self._reranker: Any = None

        # 集合字段名缓存（用于兼容缺少 doc_id 的旧集合）
        self._fields_cache: list[str] | None = None

    # ------------------------------------------------------------------ 连接
    @property
    def client(self) -> Any:
        """懒加载 MilvusClient（首次访问时建立连接）。"""
        if self._client is None:
            from pymilvus import MilvusClient

            self._client = MilvusClient(uri=self.uri, timeout=self.timeout)
            logger.info("连接 Milvus：%s", self.uri)
        return self._client

    def ping(self) -> dict[str, Any]:
        """探测 Milvus 可达性（不抛异常）。"""
        try:
            version = self.client.get_server_version()
            return {"uri": self.uri, "reachable": True, "version": version,
                    "collection": self.collection}
        except Exception as exc:  # pragma: no cover - 连接失败场景
            return {"uri": self.uri, "reachable": False, "error": str(exc)}

    # ------------------------------------------------------------------ 建表
    def _ensure_collection(self) -> None:
        """确保集合存在并加载（幂等）；旧的无 sparse 字段集合会被自动迁移。"""
        if self._collection_ready:
            return
        from pymilvus import DataType

        client = self.client
        if client.has_collection(self.collection):
            desc = client.describe_collection(self.collection)
            field_names = {str(f.get("name", "")) for f in desc.get("fields", [])}
            if self.sparse_field not in field_names:
                # 旧集合（只有稠密向量、无原生 BM25 稀疏字段）→ 自动迁移
                self._migrate_legacy()
            else:
                client.load_collection(self.collection)
                self._collection_ready = True
                logger.info("集合已存在并加载：%s", self.collection)
            return

        if self.dim is None:
            self._ensure_embedder()

        self._create_collection()
        self._collection_ready = True
        logger.info("已创建 Milvus 集合并加载：%s（dim=%s）", self.collection, self.dim)

    def _create_collection(self) -> None:
        """按「稠密 + 原生 BM25 稀疏」新 schema 创建集合并建索引（建完即加载）。"""
        from pymilvus import DataType, Function, FunctionType

        client = self.client
        schema = client.create_schema(auto_id=True, enable_dynamic_field=False)
        schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field("chunk_id", DataType.VARCHAR, max_length=128)
        schema.add_field("doc_id", DataType.VARCHAR, max_length=256)
        # text 开分析器：供 BM25 函数分词、生成稀疏向量（sparse 字段）
        schema.add_field(
            "text", DataType.VARCHAR, max_length=8192,
            enable_analyzer=True, analyzer_params={"type": self.analyzer},
        )
        schema.add_field("source", DataType.VARCHAR, max_length=512)
        schema.add_field("page", DataType.INT64)
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=self.dim)
        # 原生 BM25：sparse 字段由服务端 BM25 函数从 text 自动生成
        schema.add_field(self.sparse_field, DataType.SPARSE_FLOAT_VECTOR)
        schema.add_function(Function(
            name=BM25_FN_NAME,
            function_type=FunctionType.BM25,
            input_field_names=["text"],
            output_field_names=[self.sparse_field],
        ))

        # embedding 建 HNSW 近似最近邻索引（M 越大召回越好、efConstruction 越大建得越慢）。
        index = client.prepare_index_params()
        index.add_index(
            field_name="embedding",
            index_type="HNSW",
            metric_type=self.metric_type,
            params={"M": 16, "efConstruction": 200},
        )
        # 稀疏向量建倒排索引，metric 用 BM25（原生关键词打分）
        index.add_index(
            field_name=self.sparse_field,
            index_type=self.sparse_index_type,
            metric_type="BM25",
        )
        client.create_collection(self.collection, schema=schema, index_params=index)
        client.load_collection(self.collection)

    def _migrate_legacy(self) -> None:
        """把旧集合（仅稠密向量、无 sparse 字段）迁移成新的稀疏混合集合。

        复用旧集合里已有的 text + embedding（不重新解析/向量化）：
        先全量读出、再删旧建新、最后回插；sparse 向量由 BM25 函数自动生成。
        """
        client = self.client
        desc = client.describe_collection(self.collection)
        present = {str(f.get("name", "")) for f in desc.get("fields", [])}
        wanted = ["chunk_id", "doc_id", "text", "source", "page", "embedding"]
        fields = [f for f in wanted if f in present]

        rows: list[dict[str, Any]] = []
        offset, page = 0, 2000
        while True:
            batch = client.query(
                self.collection,
                filter="id >= 0",
                output_fields=fields,
                limit=page,
                offset=offset,
            )
            if not batch:
                break
            rows.extend(batch)
            if len(batch) < page:
                break
            offset += page

        logger.warning("迁移旧集合 %s：读出 %d 条，重建为稀疏混合集合", self.collection, len(rows))
        client.drop_collection(self.collection)
        self._create_collection()

        data = [
            {
                "chunk_id": str(r.get("chunk_id", "")),
                "doc_id": str(r.get("doc_id", "")),
                "text": str(r.get("text", "")),
                "source": str(r.get("source", "")),
                "page": int(r.get("page", 0) or 0),
                "embedding": list(r.get("embedding", [])),
            }
            for r in rows
        ]
        if data:
            client.insert(self.collection, data=data)
            client.flush(self.collection)  # 立即落盘，保证 count()/检索立刻可见
        self._collection_ready = True
        logger.info("迁移完成：%s 共 %d 条", self.collection, len(data))

    # ------------------------------------------------------------------ 向量化
    def _ensure_embedder(self) -> None:
        """确保向量化函数可用：优先用 embed_fn，否则懒加载 embed_model。"""
        if self._embed_fn is not None:
            return
        if self._embed_model is None:
            raise RuntimeError(
                "未配置 embed_fn 或 embed_model，无法进行向量检索/入库；"
                "可改用 mode='bm25' 仅做关键词检索"
            )
        from sentence_transformers import SentenceTransformer

        dev = resolve_device(self.device)
        model = SentenceTransformer(self._embed_model, device=dev)

        def _encode(texts: list[str]) -> list[list[float]]:
            vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            return [v.tolist() for v in vectors]

        self._embed_fn = _encode
        if self.dim is None:
            self.dim = int(model.get_sentence_embedding_dimension())
        logger.info("加载向量化模型：%s（device=%s，dim=%s）",
                    self._embed_model, dev, self.dim)

    def encode_query(self, query: str) -> list[float]:
        """编码单条查询为归一化向量。"""
        self._ensure_embedder()
        return self._embed_fn([query])[0]

    def encode_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """批量编码文本为归一化向量。"""
        self._ensure_embedder()
        return self._embed_fn(list(texts))

    # ------------------------------------------------------------------ 入库
    def ingest(
        self,
        texts: Sequence[str],
        vectors: Sequence[Sequence[float]] | None = None,
        metadatas: Sequence[dict[str, Any]] | None = None,
        ids: Sequence[str] | None = None,
    ) -> int:
        """批量写入 Milvus（sparse 向量由服务端 BM25 函数自动生成），返回写入条数。"""
        texts = [t[:MAX_TEXT_LEN] for t in texts]
        if not texts:
            return 0
        n = len(texts)

        if vectors is None:
            vectors = self.encode_texts(texts)
        vectors = [list(v) for v in vectors]
        if len(vectors) != n:
            raise ValueError(f"vectors 数量({len(vectors)})与 texts 数量({n})不一致")

        metadatas = list(metadatas) if metadatas is not None else [{} for _ in range(n)]
        if len(metadatas) != n:
            raise ValueError(f"metadatas 数量({len(metadatas)})与 texts 数量({n})不一致")
        ids = [str(x) for x in ids] if ids is not None else [uuid.uuid4().hex for _ in range(n)]

        # 校验向量维度并据此确定 dim
        vec_dim = len(vectors[0])
        if self.dim is None:
            self.dim = vec_dim
        elif self.dim != vec_dim:
            raise ValueError(f"向量维度 {vec_dim} 与配置 dim={self.dim} 不一致")

        with self._lock:
            self._ensure_collection()
            rows = [
                {
                    "chunk_id": ids[i],
                    "doc_id": str((metadatas[i] or {}).get("doc_id", ""))[:250],
                    "text": texts[i],
                    "source": str((metadatas[i] or {}).get("source", ""))[:500],
                    "page": int((metadatas[i] or {}).get("page", 0) or 0),
                    "embedding": vectors[i],
                }
                for i in range(n)
            ]
            self.client.insert(collection_name=self.collection, data=rows)
            logger.info("已向 Milvus 写入 %d 条（sparse 由 BM25 函数自动生成）", n)
        return n

    def _output_fields(self) -> list[str]:
        """返回集合中实际存在、且检索需要的输出字段。

        兼容由 RAG_1 等旧流程创建的集合（其 schema 缺少 ``doc_id`` 字段）：
        只请求真实存在的字段，避免 Milvus 报 ``field doc_id not exist``。
        """
        wanted = ["chunk_id", "text", "source", "page", "doc_id"]
        if self._fields_cache is None:
            self._ensure_collection()
            try:
                desc = self.client.describe_collection(self.collection)
                self._fields_cache = [str(f.get("name", "")) for f in desc.get("fields", [])]
            except Exception:  # noqa: BLE001 - 描述失败时按完整字段请求
                self._fields_cache = []
        if not self._fields_cache:
            return wanted
        return [f for f in wanted if f in self._fields_cache]

    # ------------------------------------------------------------------ 检索
    def _search_raw(self, data: Any, anns_field: str, search_params: dict, limit: int) -> list[Hit]:
        """执行一次 Milvus 单路 search 并解析；连接/检索错误返回空列表（不抛）。"""
        try:
            self._ensure_collection()
            raw = self.client.search(
                collection_name=self.collection,
                data=data,
                anns_field=anns_field,
                search_params=search_params,
                limit=limit,
                output_fields=self._output_fields(),
            )
            return _parse_hits(raw)
        except Exception as exc:
            logger.warning("Milvus 检索失败（返回空）：%s", exc)
            return []

    def search_dense(self, query: str, top_k: int = 20) -> list[Hit]:
        """Milvus 向量语义检索（单路）。缺向量化模型时抛错，Milvus 错误返回空。"""
        if self._embed_fn is None and self._embed_model is None:
            raise RuntimeError("未配置 embed_fn 或 embed_model，无法进行向量检索；可改用 mode='bm25'")
        vec = self.encode_query(query)  # 模型加载错误在此直接抛出
        # ef 是检索时的搜索宽度：越大召回越准、越慢。
        return self._search_raw([vec], "embedding",
                                {"metric_type": self.metric_type, "params": {"ef": 64}}, top_k)

    def search_bm25(self, query: str, top_k: int = 20) -> list[Hit]:
        """Milvus 原生 BM25 关键词检索（单路，直接对 sparse 字段传 query 文本）。"""
        if not query or not query.strip():
            return []
        return self._search_raw([query], self.sparse_field, {"metric_type": "BM25"}, top_k)

    # ------------------------------------------------------------------ 重排
    def _ensure_reranker(self) -> None:
        """懒加载交叉编码器重排模型。"""
        if self._reranker is not None or self._rerank_fn is not None:
            return
        if not self.rerank_model:
            raise RuntimeError(
                "未配置 rerank_model 或 rerank_fn，无法重排；"
                "请传入 rerank_model（如 'D:/modelscope/bge-reranker-v2-m3'）或 rerank_fn"
            )
        from sentence_transformers import CrossEncoder

        dev = resolve_device(self.device)
        self._reranker = CrossEncoder(
            self.rerank_model,
            max_length=self.rerank_max_length,
            device=dev,
        )
        logger.info("加载重排模型：%s（device=%s，max_length=%s）",
                    self.rerank_model, dev, self.rerank_max_length)

    def rerank(self, query: str, texts: Sequence[str]) -> list[float]:
        """用交叉编码器对候选文本打分，返回与 texts 同序的相关性分数。

        参数：
            query: 查询文本。
            texts: 候选文本列表。

        返回：
            相关性分数列表（越高越相关）。
        """
        texts = list(texts)
        if not texts:
            return []
        if self._rerank_fn is not None:
            return [float(s) for s in self._rerank_fn(query, texts)]
        self._ensure_reranker()
        pairs = [[query, t] for t in texts]
        scores = self._reranker.predict(
            pairs, batch_size=self.rerank_batch_size, show_progress_bar=False
        )
        return [float(s) for s in scores]

    def _apply_rerank(self, query: str, hits: list[Hit]) -> list[Hit]:
        """对候选命中做重排：分数替换为重排分，按降序重排并记录排名。"""
        if not hits:
            return hits
        scores = self.rerank(query, [h.text for h in hits])
        for h, s in zip(hits, scores):
            h.extra = dict(h.extra)
            h.extra["pre_rerank_score"] = round(h.score, 6)
            h.routes = dict(h.routes)
            h.score = float(s)
        hits.sort(key=lambda h: -h.score)
        for rank, h in enumerate(hits, start=1):
            h.routes["rerank"] = {"rank": float(rank), "raw": round(h.score, 6), "weight": 1.0}
        return hits

    def search(
        self,
        query: str,
        top_k: int = 10,
        mode: str = "hybrid",
        weights: dict[str, float] | None = None,
        dense_top_k: int | None = None,
        bm25_top_k: int | None = None,
        rerank: bool = False,
        rerank_top_k: int | None = None,
    ) -> list[Hit]:
        """混合检索主入口：mode=dense/bm25/hybrid；rerank=True 时二次精排。

        rerank 时 hit.score 为重排分，原融合分存 extra["pre_rerank_score"]。
        返回按最终分数降序的 Hit 列表。
        """
        query = (query or "").strip()
        if not query:
            raise ValueError("检索 query 不能为空")
        if mode not in ("dense", "bm25", "hybrid"):
            raise ValueError(f"不支持的检索模式：{mode}（可选 dense/bm25/hybrid）")

        top_k = int(top_k)
        # 重排时先召回更大的候选池（默认 3 倍 top_k），再做二次精排
        candidate_k = int(rerank_top_k or max(top_k * 3, top_k)) if rerank else top_k

        if mode == "dense":
            hits = self.search_dense(query, int(dense_top_k or candidate_k))
        elif mode == "bm25":
            hits = self.search_bm25(query, int(bm25_top_k or candidate_k))
        else:
            hits = self._search_hybrid(query, candidate_k, weights, dense_top_k, bm25_top_k)

        # per_doc_limit：每个文档最多保留 N 个片段，避免单文档霸榜
        if self.per_doc_limit > 0:
            seen: dict[str, int] = {}
            kept: list[Hit] = []
            for h in hits:
                doc_key = h.doc_id or h.source or h.chunk_id
                if seen.get(doc_key, 0) >= self.per_doc_limit:
                    continue
                seen[doc_key] = seen.get(doc_key, 0) + 1
                kept.append(h)
            hits = kept

        if rerank:
            hits = self._apply_rerank(query, hits)
        return hits[:top_k]

    def _search_hybrid(
        self,
        query: str,
        top_k: int,
        weights: dict[str, float] | None,
        dense_top_k: int | None,
        bm25_top_k: int | None,
    ) -> list[Hit]:
        """Milvus 服务端混合检索：稠密向量 + 原生 BM25 稀疏向量，融合取 top_k。"""
        from pymilvus import AnnSearchRequest, RRFRanker, WeightedRanker

        if self._embed_fn is None and self._embed_model is None:
            raise RuntimeError(
                "未配置 embed_fn 或 embed_model，无法进行混合检索；可改用 mode='bm25'"
            )
        vec = self.encode_query(query)  # 模型加载错误在此直接抛出
        dk = int(dense_top_k or top_k)
        bk = int(bm25_top_k or top_k)

        dense_req = AnnSearchRequest(
            data=[vec],
            anns_field="embedding",
            param={"metric_type": self.metric_type, "params": {"ef": 64}},
            limit=dk,
        )
        sparse_req = AnnSearchRequest(
            data=[query],
            anns_field=self.sparse_field,
            param={"metric_type": "BM25"},
            limit=bk,
        )

        dw = float((weights or {}).get("dense", self.dense_weight))
        bw = float((weights or {}).get("bm25", self.bm25_weight))
        # 两路权重相等用 RRF（只看排名，天然规避两路分数量纲不一致）；
        # 不等时改用 WeightedRanker（先归一化再按权重加权求和）。
        ranker = RRFRanker(k=self.rrf_k) if dw == bw else WeightedRanker(dw, bw)

        try:
            self._ensure_collection()
            raw = self.client.hybrid_search(
                collection_name=self.collection,
                reqs=[dense_req, sparse_req],
                ranker=ranker,
                limit=top_k,
                output_fields=self._output_fields(),
            )
        except Exception as exc:
            logger.warning("Milvus 混合检索失败（返回空）：%s", exc)
            return []

        return _parse_hits(raw)

    def search_dicts(self, query: str, top_k: int = 10, mode: str = "hybrid",
                     with_text: bool = True, **kwargs: Any) -> list[dict[str, Any]]:
        """便捷方法：返回字典列表（等价于 [h.to_dict() for h in search(...)]）。"""
        return [h.to_dict(with_text=with_text) for h in self.search(query, top_k, mode, **kwargs)]

    # ------------------------------------------------------------------ 维护
    def count(self) -> int:
        """返回集合中的实体数量。"""
        try:
            self._ensure_collection()
            stats = self.client.get_collection_stats(self.collection)
            return int(stats.get("row_count", 0))
        except Exception as exc:
            logger.warning("读取 Milvus 数量失败：%s", exc)
            return 0

    def drop(self) -> None:
        """删除集合。"""
        with self._lock:
            try:
                if self.client.has_collection(self.collection):
                    self.client.drop_collection(self.collection)
                    logger.warning("已删除集合：%s", self.collection)
                self._client = None
                self._collection_ready = False
            except Exception as exc:
                logger.warning("删除集合失败：%s", exc)


