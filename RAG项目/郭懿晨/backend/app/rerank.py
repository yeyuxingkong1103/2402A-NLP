"""粗排与精排两阶段重排。

- CoarseScorer / CoarseReranker：复用 BGE-M3 稠密信号做廉价粗排。
- FineReranker：封装 bge-reranker-v2-m3 交叉编码器做精排。
- RetrievalChain：编排召回 → 粗排 → 精排 → 阈值过滤。
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol

from backend.app.config import AppSettings
from backend.app.embeddings import TextEmbedder
from backend.app.vector_store import SearchResult, VectorStore


def _cosine(a: list[float], b: list[float]) -> float:
    """计算两个稠密向量的余弦相似度。"""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _min_max(values: list[float]) -> list[float]:
    """min-max 归一化到 [0, 1]；全等输入返回全 0。"""
    if not values:
        return []
    low, high = min(values), max(values)
    if high == low:
        return [0.0 for _ in values]
    return [(value - low) / (high - low) for value in values]


class CoarseScorer(Protocol):
    """粗排打分器协议。"""

    def score(self, query_dense: list[float] | None, results: list[SearchResult]) -> list[float]:
        """返回与 results 一一对应的粗排分。"""


class DenseCosineScorer:
    """用 BGE-M3 dense 余弦相似度打分；query 或候选 dense 缺失时回退 RRF 分。"""

    def score(self, query_dense: list[float] | None, results: list[SearchResult]) -> list[float]:
        scores: list[float] = []
        for result in results:
            if query_dense is not None and result.dense is not None:
                scores.append(_cosine(query_dense, result.dense))
            else:
                scores.append(float(result.score))
        return scores


class RrfScoreScorer:
    """直接用 RRF 融合分作为粗排分（零模型成本）。"""

    def score(self, query_dense: list[float] | None, results: list[SearchResult]) -> list[float]:
        return [float(result.score) for result in results]


class HybridScorer:
    """dense 余弦与 RRF 分归一化后加权。"""

    def __init__(self, weight: float = 0.5) -> None:
        self.weight = weight

    def score(self, query_dense: list[float] | None, results: list[SearchResult]) -> list[float]:
        dense_scores = DenseCosineScorer().score(query_dense, results)
        rrf_scores = [float(result.score) for result in results]
        dense_norm = _min_max(dense_scores)
        rrf_norm = _min_max(rrf_scores)
        return [
            self.weight * dense + (1.0 - self.weight) * rrf
            for dense, rrf in zip(dense_norm, rrf_norm, strict=True)
        ]


class CoarseReranker:
    """粗排：打分 + 降序排序 + 截断到 top_k。"""

    def __init__(self, scorer: CoarseScorer, top_k: int) -> None:
        self.scorer = scorer
        self.top_k = top_k

    def rerank(self, query_dense: list[float] | None, results: list[SearchResult]) -> list[SearchResult]:
        scores = self.scorer.score(query_dense, results)
        ranked = sorted(zip(results, scores, strict=True), key=lambda pair: pair[1], reverse=True)
        return [
            SearchResult(chunk=result.chunk, score=score, dense=result.dense)
            for result, score in ranked[: self.top_k]
        ]


_FLAGRANKER_CACHE: dict[str, object] = {}


def _resolve_reranker_dir(model_path: Path) -> Path:
    """解析重排模型目录，兼容直接模型目录与 HuggingFace 缓存布局。"""
    if (model_path / "config.json").exists():
        return model_path
    for config in sorted(model_path.rglob("config.json")):
        parent = config.parent
        if (parent / "model.safetensors").exists():
            return parent
    return model_path


def _load_flag_reranker(model_path: Path, device: str | None) -> object:
    """懒加载并缓存 FlagReranker（chat 与 eval 共享同一实例）。"""
    if not model_path.exists():
        raise FileNotFoundError(f"bge-reranker 模型路径不存在: {model_path}")
    resolved = _resolve_reranker_dir(model_path)
    cache_key = str(resolved.resolve())
    reranker = _FLAGRANKER_CACHE.get(cache_key)
    if reranker is None:
        from FlagEmbedding import FlagReranker

        use_fp16 = False
        try:
            import torch

            use_fp16 = torch.cuda.is_available()
        except Exception:
            use_fp16 = False
        reranker = FlagReranker(str(resolved), use_fp16=use_fp16)
        if device == "cpu":
            reranker.model.to("cpu")
        _FLAGRANKER_CACHE[cache_key] = reranker
    return reranker


class FineReranker:
    """精排：bge-reranker-v2-m3 交叉编码器，normalize 打分覆盖 result.score。"""

    def __init__(self, model_path: Path, batch_size: int, top_k: int, device: str | None = None) -> None:
        self.model_path = model_path
        self.batch_size = batch_size
        self.top_k = top_k
        self.device = device

    def rerank(self, query: str, results: list[SearchResult]) -> list[SearchResult]:
        if not results:
            return []
        model = _load_flag_reranker(self.model_path, self.device)
        pairs = [[query, result.chunk.text] for result in results]
        all_scores: list[float] = []
        for start in range(0, len(pairs), self.batch_size):
            batch_scores = model.compute_score(pairs[start : start + self.batch_size], normalize=True)
            if isinstance(batch_scores, float):
                batch_scores = [batch_scores]
            all_scores.extend(float(score) for score in batch_scores)
        ranked = sorted(zip(results, all_scores, strict=True), key=lambda pair: pair[1], reverse=True)
        return [
            SearchResult(chunk=result.chunk, score=score, dense=result.dense)
            for result, score in ranked[: self.top_k]
        ]


def build_coarse_scorer(settings: AppSettings) -> CoarseScorer:
    """按配置构造粗排打分器。"""
    if settings.coarse_scorer == "hybrid":
        return HybridScorer(settings.coarse_hybrid_weight)
    if settings.coarse_scorer == "dense_cosine":
        return DenseCosineScorer()
    return RrfScoreScorer()


class RetrievalChain:
    """编排召回 → 粗排 → 精排 → 阈值过滤；rerank_enabled=false 退化为 v1。"""

    def __init__(
        self,
        store: VectorStore,
        embedder: TextEmbedder,
        settings: AppSettings,
        coarse: CoarseReranker,
        fine: FineReranker,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.settings = settings
        self.coarse = coarse
        self.fine = fine

    def retrieve(self, question: str) -> list[SearchResult]:
        if not self.settings.rerank_enabled:
            results = self.store.search(question, self.embedder, limit=self.settings.final_top_k)
            return [result for result in results if result.score >= self.settings.min_retrieval_score]

        query_dense = self.embedder.embed_texts([question])[0].dense
        candidates = self.store.search(
            question,
            self.embedder,
            limit=self.settings.retrieval_candidate_k,
            with_vectors=True,
        )
        if not candidates:
            return []
        coarse_results = self.coarse.rerank(query_dense, candidates)
        if not coarse_results:
            return []
        fine_results = self.fine.rerank(question, coarse_results)
        return [result for result in fine_results if result.score >= self.settings.min_retrieval_score]
