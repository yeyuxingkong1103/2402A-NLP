# -*- coding: utf-8 -*-
"""Milvus + BM25 混合检索模块。

单文件、零硬依赖（pymilvus / jieba / sentence-transformers 均为懒加载），
提供「向量语义召回（Milvus）⊕ 关键词召回（BM25）→ 加权 RRF 融合」的混合检索能力。

设计要点：
- 稠密路：Milvus 向量检索（HNSW + COSINE），负责语义召回；
- 关键词路：内存 BM25（jieba 分词 + 自实现打分），负责精确词命中；
- 融合：两路用加权 RRF（Reciprocal Rank Fusion）合并，规避两路分数量纲不一致；
- 重排：可选加载交叉编码器（bge-reranker-v2-m3）对召回候选二次精排，提升相关性；
- 检索模式 mode：dense / bm25 / hybrid，便于做消融对比。

用法示例（详见同目录 example.py）：

    from hybrid_retriever import HybridRetriever

    def embed(texts):                       # 你的向量化函数
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer("D:/modelscope/bge-m3", device="cuda")
        return [v.tolist() for v in m.encode(texts, normalize_embeddings=True)]

    retriever = HybridRetriever(uri="http://localhost:19530",
                                collection="my_kb", dim=1024, embed_fn=embed,
                                rerank_model="D:/modelscope/bge-reranker-v2-m3")

    # 入库：写入 Milvus 的同时重建 BM25 索引
    retriever.ingest(texts=["……", "……"],
                     metadatas=[{"source": "a.pdf", "page": 1}, {}])

    # 检索：返回 List[Hit]
    hits = retriever.search("你的问题", top_k=5, mode="hybrid")

    # 检索 + 交叉编码器重排（先召回 3 倍候选，再精排取 top_k）
    hits = retriever.search("你的问题", top_k=5, mode="hybrid", rerank=True)
    for h in hits:
        print(h.chunk_id, h.score, h.text)
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

logger = logging.getLogger("rag2.hybrid_retriever")

# 文本写入 Milvus 时的最大字符数（与 VARCHAR max_length=8192 匹配，留出余量）
MAX_TEXT_LEN = 8000

# 默认重排模型（BAAI 交叉编码器，与 bge-m3 同族；sentence-transformers 懒加载）
DEFAULT_RERANK_MODEL = r"D:\modelscope\bge-reranker-v2-m3"

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
    """中英混合分词：jieba 精确模式 + 停用词/标点过滤 + 小写化。

    参数：
        text:      输入文本。
        stopwords: 停用词集合。

    返回：
        词语列表。
    """
    import jieba

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
        """转换为字典。

        参数：
            with_text:  是否附带文本内容。
            text_limit: 文本截断长度（超出追加省略号）。

        返回：
            结果字典。
        """
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
    """加权 RRF（Reciprocal Rank Fusion）融合。

    RRF 只看排名不看原始分，天然规避稠密余弦、BM25 两路量纲不一致的问题：

        score(d) = Σ_r  weight_r / (k + rank_r(d))

    其中 rank 从 1 开始；k 为平滑常数（默认 60，与 Milvus 原生 RRFRanker 一致）。

    参数：
        rankings: {路名: [chunk_id, ...]}，各路按相关性从高到低排序。
        weights:  {路名: 权重}，缺省 1.0；权重 <= 0 的路直接忽略。
        k:        RRF 平滑常数，必须 >= 1。
        scores:   {路名: {chunk_id: 原始分}}，仅用于回填调试信息。

    返回：
        按融合分数降序排列的 FusedItem 列表。
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
        """初始化 BM25 超参数。

        参数：
            k1:        词频饱和参数。
            b:         文档长度归一化参数。
            stopwords: 停用词集合；缺省使用内置中文/英文停用词。
            tokenizer: 自定义分词函数；缺省使用 jieba 分词。
        """
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
        """对文本列表分词并构建索引。

        参数：
            texts: 文档文本列表。
        """
        self.docs = [self._tokenizer(t) for t in texts]
        self.doc_len = [len(d) for d in self.docs]
        n = len(self.docs)
        self.avg_len = sum(self.doc_len) / n if n else 0.0

        df: dict[str, int] = {}
        for doc in self.docs:
            for term in set(doc):
                df[term] = df.get(term, 0) + 1
        self.df = df

        # IDF：df 越高，区分度越低
        self.idf = {term: math.log(1 + (n - cnt + 0.5) / (cnt + 0.5))
                    for term, cnt in df.items()}
        self._built = True
        logger.info("BM25 索引构建完成：文档数=%d", n)

    def search(self, query: str, top_k: int = 20) -> list[tuple[int, float]]:
        """按 BM25 分数检索，返回 (文档下标, 分数) 列表。

        参数：
            query: 查询文本。
            top_k: 返回候选数量。

        返回：
            按分数降序排列的 (文档下标, 分数) 元组列表。
        """
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
            denom = f + self.k1 * (1 - self.b + self.b * doc_len / self.avg_len)
            total += idf * (f * (self.k1 + 1)) / denom
        return total


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
        device: str = "cpu",
        k1: float = 1.5,
        b: float = 0.75,
        stopwords: Iterable[str] | None = None,
        tokenizer: Callable[[str], list[str]] | None = None,
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
        """初始化混合检索器（懒连接：首次入库/检索时才连接 Milvus）。

        参数：
            uri:          Milvus 服务地址。
            collection:   Milvus 集合名。
            dim:          向量维度；缺省 1024（bge-m3 dense 维度）。
                          使用 embed_model 且传 None 时自动从模型推断。
            metric_type:  距离度量（COSINE / L2 / IP）。
            embed_fn:     向量化函数，输入文本列表，输出同序向量列表。
            embed_model:  sentence-transformers 模型名/路径；未提供 embed_fn
                          时据此懒加载模型（自动归一化）。
            device:       embed_model 加载设备（cuda / cpu）。
            k1 / b:       BM25 超参数。
            stopwords:    BM25 停用词；缺省内置中英停用词。
            tokenizer:    BM25 自定义分词；缺省 jieba。
            rrf_k:        RRF 平滑常数。
            dense_weight: hybrid 模式中稠密路的融合权重。
            bm25_weight:  hybrid 模式中关键词路的融合权重。
            per_doc_limit: 每个文档最多保留的片段数（0 表示不限），避免单文档霸榜。
            timeout:      Milvus 客户端超时（秒）。
            rerank_model: 交叉编码器重排模型路径/名（如 bge-reranker-v2-m3）；
                          缺省 None，不启用重排，需 search(rerank=True) 时才加载。
            rerank_fn:    自定义重排函数 (query, texts) -> 与 texts 同序的相关性分数；
                          提供后优先于 rerank_model。
            rerank_batch_size: 重排打分批大小。
            rerank_max_length: 重排模型最大输入长度（token 数）。
        """
        self.uri = uri
        self.collection = collection
        self.dim = int(dim) if dim else None
        self.metric_type = metric_type
        self._embed_fn = embed_fn
        self._embed_model = embed_model
        self.device = device
        self.k1 = k1
        self.b = b
        self._stopwords = frozenset(stopwords) if stopwords is not None else _DEFAULT_STOPWORDS
        self._tokenizer = tokenizer
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

        # BM25 语料（与 Milvus 中的 chunk_id 一一对应，供融合对齐）
        self._bm25: BM25Index | None = None
        self._bm25_texts: list[str] = []
        self._bm25_ids: list[str] = []
        self._bm25_metas: list[dict[str, Any]] = []

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
        """创建集合与索引（若尚不存在，幂等）。"""
        if self._collection_ready:
            return
        from pymilvus import DataType

        client = self.client
        if client.has_collection(self.collection):
            client.load_collection(self.collection)
            self._collection_ready = True
            return

        if self.dim is None:
            self._ensure_embedder()

        schema = client.create_schema(auto_id=True, enable_dynamic_field=False)
        schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field("chunk_id", DataType.VARCHAR, max_length=128)
        schema.add_field("doc_id", DataType.VARCHAR, max_length=256)
        schema.add_field("text", DataType.VARCHAR, max_length=8192)
        schema.add_field("source", DataType.VARCHAR, max_length=512)
        schema.add_field("page", DataType.INT64)
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=self.dim)

        index = client.prepare_index_params()
        index.add_index(
            field_name="embedding",
            index_type="HNSW",
            metric_type=self.metric_type,
            params={"M": 16, "efConstruction": 200},
        )
        client.create_collection(self.collection, schema=schema, index_params=index)
        client.load_collection(self.collection)
        self._collection_ready = True
        logger.info("已创建 Milvus 集合并加载：%s（dim=%s）", self.collection, self.dim)

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

        model = SentenceTransformer(self._embed_model, device=self.device)

        def _encode(texts: list[str]) -> list[list[float]]:
            vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            return [v.tolist() for v in vectors]

        self._embed_fn = _encode
        if self.dim is None:
            self.dim = int(model.get_sentence_embedding_dimension())
        logger.info("加载向量化模型：%s（device=%s，dim=%s）",
                    self._embed_model, self.device, self.dim)

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
        """写入文本：批量写入 Milvus 并重建 BM25 索引。

        参数：
            texts:      分块文本列表。
            vectors:    与 texts 对应的向量列表；缺省时用 embed_fn 自动编码。
            metadatas:  与 texts 对应的元数据列表，支持 doc_id/source/page 及任意自定义键。
            ids:        chunk_id 列表（融合对齐键）；缺省自动生成。

        返回：
            写入条数。
        """
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
            # 写入 Milvus（失败不阻断 BM25，只告警降级）
            try:
                self._ensure_collection()
                rows = []
                for i in range(n):
                    meta = metadatas[i] or {}
                    rows.append({
                        "chunk_id": ids[i],
                        "doc_id": str(meta.get("doc_id", ""))[:250],
                        "text": texts[i],
                        "source": str(meta.get("source", ""))[:500],
                        "page": int(meta.get("page", 0) or 0),
                        "embedding": vectors[i],
                    })
                self.client.insert(collection_name=self.collection, data=rows)
                logger.info("已向 Milvus 写入 %d 条", n)
            except Exception as exc:
                logger.warning("写入 Milvus 失败（BM25 索引仍会更新，仅剩关键词路可用）：%s", exc)

            # 更新内存 BM25 语料并重建
            self._bm25_texts.extend(texts)
            self._bm25_ids.extend(ids)
            self._bm25_metas.extend(metadatas)
            self._rebuild_bm25()
        return n

    def _rebuild_bm25(self) -> None:
        self._bm25 = BM25Index(k1=self.k1, b=self.b,
                               stopwords=self._stopwords, tokenizer=self._tokenizer)
        self._bm25.build(self._bm25_texts)

    def rebuild_from_milvus(self) -> int:
        """从已存在的 Milvus 集合全量拉取分块，重建 BM25 索引。

        适用于：对已有集合新建检索器实例、或外部直接写入了 Milvus 的场景。

        返回：
            加载的分块数量。
        """
        self._ensure_collection()
        rows: list[dict[str, Any]] = []
        offset, page = 0, 2000
        while True:
            batch = self.client.query(
                self.collection,
                filter="id >= 0",
                output_fields=self._output_fields(),
                limit=page,
                offset=offset,
            )
            if not batch:
                break
            rows.extend(batch)
            if len(batch) < page:
                break
            offset += page

        with self._lock:
            self._bm25_texts = [str(r.get("text", "")) for r in rows]
            self._bm25_ids = [str(r.get("chunk_id", "")) for r in rows]
            self._bm25_metas = [
                {"doc_id": str(r.get("doc_id", "")), "source": str(r.get("source", "")),
                 "page": int(r.get("page", 0) or 0)}
                for r in rows
            ]
            self._rebuild_bm25()
        logger.info("已从 Milvus 重建 BM25 索引（%d 条）", len(rows))
        return len(rows)

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
    def search_dense(self, query: str, top_k: int = 20) -> list[Hit]:
        """Milvus 向量语义检索（单路）。

        向量化模型加载错误会直接抛出；Milvus 连接/检索错误则告警并返回空列表。
        """
        if self._embed_fn is None and self._embed_model is None:
            raise RuntimeError(
                "未配置 embed_fn 或 embed_model，无法进行向量检索；可改用 mode='bm25'"
            )
        vec = self.encode_query(query)  # 模型加载错误在此直接抛出
        hits: list[Hit] = []
        try:
            self._ensure_collection()
            raw = self.client.search(
                collection_name=self.collection,
                data=[vec],
                anns_field="embedding",
                search_params={"metric_type": self.metric_type, "params": {"ef": 64}},
                limit=top_k,
                output_fields=self._output_fields(),
            )
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
        except Exception as exc:
            logger.warning("Milvus 向量检索失败（返回空，仅剩关键词路）：%s", exc)
        return hits

    def search_bm25(self, query: str, top_k: int = 20) -> list[Hit]:
        """BM25 关键词检索（单路）。"""
        if self._bm25 is None or not self._bm25.built:
            return []
        hits: list[Hit] = []
        for idx, score in self._bm25.search(query, top_k):
            meta = self._bm25_metas[idx] if idx < len(self._bm25_metas) else {}
            hits.append(Hit(
                chunk_id=self._bm25_ids[idx],
                text=self._bm25_texts[idx],
                doc_id=str(meta.get("doc_id", "")),
                source=str(meta.get("source", "")),
                page=int(meta.get("page", 0) or 0),
                score=score,
            ))
        return hits

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

        self._reranker = CrossEncoder(
            self.rerank_model,
            max_length=self.rerank_max_length,
            device=self.device,
        )
        logger.info("加载重排模型：%s（device=%s，max_length=%s）",
                    self.rerank_model, self.device, self.rerank_max_length)

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
        """混合检索主入口。

        参数：
            query:       查询文本。
            top_k:       最终返回的片段数量。
            mode:        检索模式 dense / bm25 / hybrid。
            weights:     融合权重，如 {"dense": 1.0, "bm25": 1.0}；
                         缺省取构造参数 dense_weight / bm25_weight。
            dense_top_k: 稠密路召回候选数量（缺省等于候选池大小）。
            bm25_top_k:  关键词路召回候选数量（缺省等于候选池大小）。
            rerank:      是否用交叉编码器对候选二次精排；为 True 时
                         hit.score 为重排分数，原融合分存于 extra["pre_rerank_score"]。
            rerank_top_k: 进入重排的候选池大小；缺省 max(top_k * 3, top_k)。

        返回：
            按最终分数降序排列的 Hit 列表。
        """
        query = (query or "").strip()
        if not query:
            raise ValueError("检索 query 不能为空")
        if mode not in ("dense", "bm25", "hybrid"):
            raise ValueError(f"不支持的检索模式：{mode}（可选 dense/bm25/hybrid）")

        top_k = int(top_k)
        # 重排时先召回更大的候选池（默认 3 倍 top_k），再做二次精排
        candidate_k = int(rerank_top_k or max(top_k * 3, top_k)) if rerank else top_k
        dk = int(dense_top_k or candidate_k)
        bk = int(bm25_top_k or candidate_k)

        rankings: dict[str, list[str]] = {}
        raw_scores: dict[str, dict[str, float]] = {}
        payloads: dict[str, Hit] = {}

        def _collect(route: str, hits: list[Hit]) -> None:
            order: list[str] = []
            smap: dict[str, float] = {}
            for h in hits:
                if h.chunk_id in smap:
                    continue
                smap[h.chunk_id] = h.score
                order.append(h.chunk_id)
                payloads.setdefault(h.chunk_id, h)
            if order:
                rankings[route] = order
                raw_scores[route] = smap

        if mode in ("dense", "hybrid"):
            _collect("dense", self.search_dense(query, dk))
        if mode in ("bm25", "hybrid"):
            _collect("bm25", self.search_bm25(query, bk))

        if not rankings:
            return []

        if weights is None:
            if mode == "dense":
                weights = {"dense": 1.0}
            elif mode == "bm25":
                weights = {"bm25": 1.0}
            else:
                weights = {"dense": self.dense_weight, "bm25": self.bm25_weight}

        # 单路：直接按原始分排序；多路：加权 RRF 融合
        if len(rankings) == 1:
            route = next(iter(rankings))
            w = float(weights.get(route, 1.0))
            fused = [
                (cid, raw_scores[route][cid],
                 {route: {"rank": float(r), "raw": round(raw_scores[route][cid], 6), "weight": w}})
                for r, cid in enumerate(rankings[route], start=1)
            ]
            fused.sort(key=lambda t: -t[1])
        else:
            items = weighted_rrf(rankings, weights=weights, k=self.rrf_k, scores=raw_scores)
            fused = [(it.key, it.score, it.routes) for it in items]

        results: list[Hit] = []
        doc_counter: dict[str, int] = {}
        for cid, score, routes in fused:
            hit = payloads.get(cid)
            if hit is None:
                continue
            if self.per_doc_limit > 0:
                doc_key = hit.doc_id or hit.source or cid
                if doc_counter.get(doc_key, 0) >= self.per_doc_limit:
                    continue
                doc_counter[doc_key] = doc_counter.get(doc_key, 0) + 1
            results.append(Hit(
                chunk_id=cid, text=hit.text, score=score,
                doc_id=hit.doc_id, source=hit.source, page=hit.page,
                routes=routes, extra=hit.extra,
            ))
            if len(results) >= candidate_k:
                break

        if rerank:
            results = self._apply_rerank(query, results)
        return results[:top_k]

    def search_dicts(self, query: str, top_k: int = 10, mode: str = "hybrid",
                     with_text: bool = True, **kwargs: Any) -> list[dict[str, Any]]:
        """便捷方法：返回字典列表（等价于 [h.to_dict() for h in search(...)]）。"""
        return [h.to_dict(with_text=with_text) for h in self.search(query, top_k, mode, **kwargs)]

    # ------------------------------------------------------------------ 维护
    def count(self) -> int:
        """返回集合中的实体数量（Milvus 不可用时回退为内存 BM25 语料数）。"""
        try:
            self._ensure_collection()
            stats = self.client.get_collection_stats(self.collection)
            return int(stats.get("row_count", 0))
        except Exception as exc:
            logger.warning("读取 Milvus 数量失败，回退为内存语料数：%s", exc)
            return len(self._bm25_texts)

    def drop(self) -> None:
        """删除集合并清空本地 BM25 索引。"""
        with self._lock:
            try:
                if self.client.has_collection(self.collection):
                    self.client.drop_collection(self.collection)
                    logger.warning("已删除集合：%s", self.collection)
                self._client = None
                self._collection_ready = False
            except Exception as exc:
                logger.warning("删除集合失败：%s", exc)
            self._bm25_texts = []
            self._bm25_ids = []
            self._bm25_metas = []
            self._bm25 = None


if __name__ == "__main__":  # pragma: no cover - 无需 Milvus/模型即可跑通的冒烟演示
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s - %(message)s")

    # 用字符袋向量作为 toy 语义向量，演示流程（真实场景请用 bge-m3 等）
    DIM = 64

    def fake_embed(texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for t in texts:
            v = [0.0] * DIM
            for ch in t:
                v[ord(ch) % DIM] += 1.0
            norm = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / norm for x in v])
        return out

    retriever = HybridRetriever(uri="http://localhost:19530", collection="demo_kb",
                                dim=DIM, embed_fn=fake_embed)
    retriever.ingest(
        texts=[
            "苹果富含维生素 C，有助于增强免疫力",
            "香蕉富含钾元素，适合运动后食用",
            "胡萝卜富含胡萝卜素，对眼睛有益",
        ],
        metadatas=[
            {"source": "fruit.pdf", "page": 1},
            {"source": "fruit.pdf", "page": 2},
            {"source": "veg.pdf", "page": 1},
        ],
    )

    for mode in ("bm25", "dense", "hybrid"):
        print(f"\n===== mode={mode} =====")
        for h in retriever.search("哪种水果富含钾元素", top_k=3, mode=mode):
            print(h.to_dict(with_text=True, text_limit=40))
