"""
retrieval.py — 混合检索与重排序

流程（对应需求文档 4.4）：多路召回（向量 + BM25 + Redis 缓存）→ 融合补分
→ BGE-rerank 重排取 Top-K → 余弦相似度过滤。

召回与融合交给 retrieval_lc：三路都包装成 LangChain 的 BaseRetriever，
由 EnsembleRetriever 做加权 RRF 融合。本模块只负责 BM25 索引的构建缓存、
以及检索后的补分、重排与阈值过滤。

BM25 用 jieba 分词在内存自建索引，按集合缓存，条数变化时自动重建，
LOCAL_MODE 与 Milvus 模式下行为一致。
"""

from __future__ import annotations

import math
import threading
from typing import Any

import config
import embeddings
import retrieval_lc
from rerank_filter import cosine_similarity
from reranker import rerank as rerank_docs
from vector_store import get_store

# BM25 建索引时最多拉取的文档数，超出部分不参与关键词召回
MAX_INDEX_DOCS = 20000


def _tokenize(text: str) -> list[str]:
    """中英文分词，用于 BM25。jieba 缺失时退化为按字符切分。"""
    text = (text or "").lower().strip()
    if not text:
        return []
    try:
        import jieba

        tokens = [w.strip() for w in jieba.lcut(text)]
    except ImportError:
        tokens = list(text)
    return [t for t in tokens if t and not t.isspace()]


class BM25Index:
    """标准 BM25（Okapi）内存索引。"""

    def __init__(self, documents: list[dict], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents = [dict(doc) for doc in documents]
        self.corpus_tokens = [_tokenize(doc.get("text", "")) for doc in self.documents]

        self.doc_count = len(self.documents)
        self.avg_length = (
            sum(len(tokens) for tokens in self.corpus_tokens) / self.doc_count
            if self.doc_count
            else 0.0
        )

        # 文档频率：某个词出现在多少个文档里
        self.doc_freq: dict[str, int] = {}
        self.term_freq: list[dict[str, int]] = []
        for tokens in self.corpus_tokens:
            counts: dict[str, int] = {}
            for token in tokens:
                counts[token] = counts.get(token, 0) + 1
            self.term_freq.append(counts)
            for token in counts:
                self.doc_freq[token] = self.doc_freq.get(token, 0) + 1

    def _idf(self, term: str) -> float:
        freq = self.doc_freq.get(term, 0)
        if freq == 0:
            return 0.0
        return math.log((self.doc_count - freq + 0.5) / (freq + 0.5) + 1.0)

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        """返回按 BM25 分数降序的文档，每条带 bm25_score 字段。"""
        query_tokens = _tokenize(query)
        if not query_tokens or not self.documents:
            return []

        scores = [0.0] * self.doc_count
        for term in set(query_tokens):
            idf = self._idf(term)
            if idf == 0.0:
                continue
            for idx in range(self.doc_count):
                freq = self.term_freq[idx].get(term, 0)
                if freq == 0:
                    continue
                length = len(self.corpus_tokens[idx]) or 1
                denominator = freq + self.k1 * (1 - self.b + self.b * length / (self.avg_length or 1))
                scores[idx] += idf * freq * (self.k1 + 1) / denominator

        ranked = sorted(range(self.doc_count), key=lambda i: -scores[i])[:top_k]
        hits: list[dict] = []
        for idx in ranked:
            if scores[idx] <= 0:
                continue
            item = {k: v for k, v in self.documents[idx].items() if k != "embedding"}
            item["bm25_score"] = float(scores[idx])
            hits.append(item)
        return hits


# ---------------------------------------------------------------- 索引缓存
_index_cache: dict[str, tuple[int, BM25Index]] = {}
_index_lock = threading.Lock()


def _fetch_all(collection: str, batch: int = 1000) -> list[dict]:
    """分页拉取集合内的全部记录，用于构造 BM25 索引。"""
    store = get_store()
    documents: list[dict] = []
    offset = 0

    while len(documents) < MAX_INDEX_DOCS:
        page = store.query(collection, limit=batch, offset=offset)
        if not page:
            break
        documents.extend(page)
        offset += len(page)
        if len(page) < batch:
            break
    return documents


def build_bm25_index(collection: str, force: bool = False) -> BM25Index:
    """构造（或复用缓存的）BM25 索引。集合条数变化时自动重建。"""
    store = get_store()
    count = store.count(collection)

    with _index_lock:
        cached = _index_cache.get(collection)
        if cached and cached[0] == count and not force:
            return cached[1]

    documents = _fetch_all(collection)
    index = BM25Index(documents)
    with _index_lock:
        _index_cache[collection] = (count, index)
    return index


def invalidate_index(collection: str | None = None) -> None:
    """清空 BM25 索引缓存。新数据导入后调用，下次检索会重建。"""
    with _index_lock:
        if collection is None:
            _index_cache.clear()
        else:
            _index_cache.pop(collection, None)


# ---------------------------------------------------------------- 混合检索
def _backfill_vector_scores(query: str, candidates: list[dict]) -> None:
    """给只有 BM25 命中的候选补算向量相似度。

    这些候选从关键词路来，metadata 里的 vector_score 是 None；不补算的话
    下面的余弦阈值会把它们当 0 分全部丢掉。
    """
    pending = [c for c in candidates if c.get("vector_score") is None]
    if not pending:
        return

    query_vector = embeddings.encode_query(query)
    vectors = embeddings.encode_texts([c.get("text", "") for c in pending])
    for candidate, vector in zip(pending, vectors):
        candidate["vector_score"] = cosine_similarity(query_vector, vector)


def _score_by_rank(candidates: list[dict]) -> None:
    """把融合名次归一化成 0~1 写进 score，供前端展示相关度。

    候选已按 RRF 分数降序排列，这里只做单调映射；开了重排时排序由
    rerank_score 决定，score 仅作兜底展示。
    """
    total = len(candidates)
    for rank, candidate in enumerate(candidates, start=1):
        candidate["score"] = round(1.0 - (rank - 1) / total, 6)


def hybrid_retrieve(
    query: str,
    collection_name: str,
    top_k: int | None = None,
    filters: dict[str, Any] | None = None,
    use_rerank: bool = True,
    threshold: float | None = None,
) -> list[dict]:
    """混合检索主入口，返回可直接交给大模型的片段列表。"""
    query = (query or "").strip()
    if not query:
        return []

    top_k = config.RERANK_TOP_K if top_k is None else top_k
    # 「余弦相似度低于 0.3 丢弃」只对真实语义向量成立：本地 hash 后端是特征哈希，
    # 分数不代表语义相似度，套用会把正确结果全丢掉；调用方显式传入 threshold 时以其为准。
    if threshold is None:
        threshold = None if config.EMBEDDING_BACKEND == "hash" else config.SIMILARITY_THRESHOLD

    # 1. 缓存前置命中：同一集合下的同一问题直接复用上次结果，检索与重排都省掉
    cached = retrieval_lc.cache_get(collection_name, query)
    if cached:
        return cached[:top_k]

    # 2. 多路召回：向量 + BM25 + 缓存三路由 EnsembleRetriever 做加权 RRF 融合
    documents = retrieval_lc.build_ensemble(collection_name, filters).invoke(query)
    candidates = [retrieval_lc.from_document(doc) for doc in documents]
    if not candidates:
        return []

    # 3. 补分与展示分
    _backfill_vector_scores(query, candidates)
    _score_by_rank(candidates)

    # 4. 重排
    if use_rerank:
        candidates = rerank_docs(query, candidates, top_k=max(top_k, config.RERANK_TOP_K))

    # 5. 余弦相似度过滤（用向量分数，重排分数只决定顺序）
    kept = candidates if threshold is None else [
        c for c in candidates if float(c.get("vector_score") or 0.0) >= threshold
    ]
    result = kept[:top_k]
    retrieval_lc.cache_put(collection_name, query, result)
    return result


if __name__ == "__main__":
    import knowledge_base
    from vector_store import reset_store

    reset_store()
    kb = knowledge_base.KnowledgeBase()
    kb.import_texts(
        [
            "劳动合同解除时，用人单位应当向劳动者支付经济补偿。",
            "经济补偿按劳动者在本单位工作的年限，每满一年支付一个月工资。",
            "今天天气晴朗，适合外出散步和郊游。",
            "劳动者提前三十日以书面形式通知用人单位，可以解除劳动合同。",
        ],
        domain="legal", source="self_test.txt",
    )
    results = hybrid_retrieve("劳动合同解除的赔偿标准", "kb_legal", top_k=3)
    for item in results:
        print(f"  融合分={item['score']:.3f} | {item['text'][:24]}")

    assert results and "天气" not in results[0]["text"], "检索结果不合理"
    print("retrieval 自检通过。")
