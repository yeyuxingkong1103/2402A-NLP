from pathlib import Path

import pytest

from backend.app.models import Chunk
from backend.app.rerank import (
    CoarseReranker,
    DenseCosineScorer,
    FineReranker,
    HybridScorer,
    RetrievalChain,
    RrfScoreScorer,
)
from backend.app.vector_store import SearchResult


def make_result(chunk_id: str, text: str, score: float, dense: list[float] | None) -> SearchResult:
    return SearchResult(
        chunk=Chunk(
            chunk_id=chunk_id,
            document_id="doc-1",
            page=1,
            category="正文",
            text=text,
            source_span="page=1:block=1",
        ),
        score=score,
        dense=dense,
    )


def test_dense_cosine_scorer_uses_query_dense():
    scorer = DenseCosineScorer()
    results = [
        make_result("a", "第一段", 0.3, [1.0, 0.0]),
        make_result("b", "第二段", 0.5, [0.0, 1.0]),
    ]

    scores = scorer.score([1.0, 0.0], results)

    assert scores[0] > scores[1]


def test_dense_cosine_scorer_falls_back_to_rrf_when_no_dense():
    scorer = DenseCosineScorer()
    results = [make_result("a", "第一段", 0.42, None)]

    scores = scorer.score(None, results)

    assert scores == [0.42]


def test_rrf_score_scorer_returns_scores():
    scorer = RrfScoreScorer()
    results = [make_result("a", "第一段", 0.4, None), make_result("b", "第二段", 0.9, None)]

    assert scorer.score(None, results) == [0.4, 0.9]


def test_hybrid_scorer_weighted():
    scorer = HybridScorer(weight=0.5)
    results = [
        make_result("a", "第一段", 1.0, [1.0, 0.0]),
        make_result("b", "第二段", 0.0, [0.0, 1.0]),
    ]

    scores = scorer.score([1.0, 0.0], results)

    assert scores[0] > scores[1]


def test_coarse_reranker_sorts_and_truncates():
    reranker = CoarseReranker(DenseCosineScorer(), top_k=1)
    results = [
        make_result("a", "第一段", 0.1, [1.0, 0.0]),
        make_result("b", "第二段", 0.9, [0.0, 1.0]),
    ]

    kept = reranker.rerank([1.0, 0.0], results)

    assert [item.chunk.chunk_id for item in kept] == ["a"]


class FakeFlagReranker:
    def __init__(self, scores: list[float]) -> None:
        self.scores = scores

    def compute_score(self, pairs, normalize: bool = False):
        return self.scores[: len(pairs)]


def test_fine_reranker_scores_and_takes_top_k(monkeypatch):
    import backend.app.rerank as rerank_module

    monkeypatch.setattr(rerank_module, "_load_flag_reranker", lambda path, device: FakeFlagReranker([0.9, 0.2, 0.5]))

    fine = FineReranker(Path("models/bge-reranker-v2-m3"), batch_size=2, top_k=2)
    results = [
        make_result("a", "第一段", 1.0, None),
        make_result("b", "第二段", 1.0, None),
        make_result("c", "第三段", 1.0, None),
    ]

    kept = fine.rerank("问题", results)

    assert [item.chunk.chunk_id for item in kept] == ["a", "c"]
    assert kept[0].score == 0.9


def test_fine_reranker_raises_when_model_missing():
    fine = FineReranker(Path("nonexistent/path"), batch_size=2, top_k=1)

    with pytest.raises(FileNotFoundError, match="bge-reranker"):
        fine.rerank("问题", [make_result("a", "第一段", 1.0, None)])


class FakeSettingsForChain:
    rerank_enabled = True
    retrieval_candidate_k = 30
    coarse_top_k = 10
    final_top_k = 6
    min_retrieval_score = 0.2


class FakeChainStore:
    def __init__(self) -> None:
        self.results = [
            make_result("a", "第一段", 0.9, [1.0, 0.0]),
            make_result("b", "第二段", 0.1, [0.0, 1.0]),
        ]

    def search(self, query, embedder, limit, with_vectors=False):
        return list(self.results)


class FakeChainEmbedder:
    def embed_texts(self, texts):
        from backend.app.embeddings import EmbeddingResult

        return [EmbeddingResult(dense=[1.0, 0.0], sparse_indices=[0], sparse_values=[1.0]) for _ in texts]


def _make_chain(store):
    return RetrievalChain(
        store,
        FakeChainEmbedder(),
        FakeSettingsForChain(),
        CoarseReranker(DenseCosineScorer(), top_k=10),
        FineReranker(Path("models/bge-reranker-v2-m3"), batch_size=2, top_k=6),
    )


def test_retrieval_chain_runs_v2_path(monkeypatch):
    import backend.app.rerank as rerank_module

    monkeypatch.setattr(rerank_module, "_load_flag_reranker", lambda path, device: FakeFlagReranker([0.9, 0.05]))

    results = _make_chain(FakeChainStore()).retrieve("问题")

    assert [item.chunk.chunk_id for item in results] == ["a"]
    assert results[0].score == 0.9


def test_retrieval_chain_v1_path_when_rerank_disabled(monkeypatch):
    import backend.app.rerank as rerank_module

    monkeypatch.setattr(rerank_module, "_load_flag_reranker", lambda path, device: FakeFlagReranker([0.9, 0.05]))

    settings = FakeSettingsForChain()
    settings.rerank_enabled = False
    chain = RetrievalChain(FakeChainStore(), FakeChainEmbedder(), settings, None, None)

    results = chain.retrieve("问题")

    assert [item.chunk.chunk_id for item in results] == ["a"]


def test_retrieval_chain_returns_empty_when_no_candidates(monkeypatch):
    import backend.app.rerank as rerank_module

    monkeypatch.setattr(rerank_module, "_load_flag_reranker", lambda path, device: FakeFlagReranker([]))

    store = FakeChainStore()
    store.results = []

    results = _make_chain(store).retrieve("问题")

    assert results == []


def test_resolve_reranker_dir_flat(tmp_path):
    from backend.app.rerank import _resolve_reranker_dir

    (tmp_path / "config.json").write_text("{}", encoding="utf-8")

    assert _resolve_reranker_dir(tmp_path) == tmp_path


def test_resolve_reranker_dir_hf_cache(tmp_path):
    from backend.app.rerank import _resolve_reranker_dir

    model_dir = tmp_path / "models" / "BAAI--bge-reranker-v2-m3" / "snapshots" / "master"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "model.safetensors").write_bytes(b"x")

    assert _resolve_reranker_dir(tmp_path) == model_dir
