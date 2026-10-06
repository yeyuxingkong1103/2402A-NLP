from unittest.mock import Mock

from src.edu_rag_ingest.retrieval.reranker import CrossEncoderReranker, RerankConfig


class Candidate:
    def __init__(self, content: str) -> None:
        self.content = content


def test_disabled_reranker_keeps_candidate_order():
    reranker = CrossEncoderReranker(RerankConfig(False, "", "cpu", 8, 10))
    candidates = [Candidate("first"), Candidate("second")]

    assert reranker.rerank("query", candidates, 1) == candidates[:1]


def test_reranker_sorts_by_model_score():
    reranker = object.__new__(CrossEncoderReranker)
    reranker.config = RerankConfig(True, "model", "cpu", 8, 10)
    reranker.model = Mock()
    reranker.model.predict.return_value = [0.2, 0.9]
    candidates = [Candidate("first"), Candidate("second")]

    results = reranker.rerank("query", candidates, 2)

    assert [item.content for item in results] == ["second", "first"]
    reranker.model.predict.assert_called_once_with(
        [["query", "first"], ["query", "second"]], batch_size=8
    )


def test_reranker_failure_falls_back_to_original_order():
    reranker = object.__new__(CrossEncoderReranker)
    reranker.config = RerankConfig(True, "model", "cpu", 8, 10)
    reranker.model = Mock()
    reranker.model.predict.side_effect = RuntimeError("inference failed")
    candidates = [Candidate("first"), Candidate("second")]

    assert reranker.rerank("query", candidates, 2) == candidates
