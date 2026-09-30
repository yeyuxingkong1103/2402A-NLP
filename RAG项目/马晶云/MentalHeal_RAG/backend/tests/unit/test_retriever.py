from types import SimpleNamespace

from app.rag.retriever import KnowledgeRetriever


def make_candidate(
    chunk_id: str,
    text: str,
    *,
    document_id: str = "doc-a",
    page_start: int = 1,
    score: float = 0.8,
    rerank_score: float = 0.8,
    combined_score: float | None = None,
) -> dict:
    candidate = {
        "chunk_id": chunk_id,
        "document_id": document_id,
        "title": f"{document_id}.pdf",
        "page_start": page_start,
        "page_end": page_start,
        "text": text,
        "score": score,
        "rerank_score": rerank_score,
    }
    if combined_score is not None:
        candidate["combined_score"] = combined_score
    return candidate


def test_lexical_score_prefers_topic_matching_text() -> None:
    query = "睡眠不好怎么办"
    relevant = KnowledgeRetriever._lexical_score(query, "可以固定作息，建立睡前放松习惯，帮助改善睡眠和入睡困难。")
    unrelated = KnowledgeRetriever._lexical_score(query, "可以识别压力来源，并与可信任的人交流。")

    assert relevant > unrelated
    assert relevant > 0


def test_lexical_score_ignores_whitespace() -> None:
    assert KnowledgeRetriever._lexical_score("不想 活", "如果有人不想活，应保持陪伴并评估危险") > 0


def test_select_candidates_removes_duplicate_chunks_and_text() -> None:
    candidates = [
        make_candidate("best", "重复内容", combined_score=1.0),
        make_candidate("same-text", "重复 内容", combined_score=0.9),
        make_candidate("other", "另一段内容", document_id="doc-b", combined_score=0.8),
    ]

    selected = KnowledgeRetriever._select_candidates(candidates, 3)

    assert [item["chunk_id"] for item in selected] == ["best", "other"]


def test_select_candidates_limits_document_and_page_concentration() -> None:
    candidates = [
        make_candidate("a1", "第一段", combined_score=1.0),
        make_candidate("a2", "第二段", combined_score=0.9),
        make_candidate("a3", "第三段", combined_score=0.8),
        make_candidate("b1", "其他来源", document_id="doc-b", combined_score=0.7),
    ]

    selected = KnowledgeRetriever._select_candidates(
        candidates,
        3,
        max_chunks_per_document=2,
        max_chunks_per_page=1,
    )

    assert [item["chunk_id"] for item in selected] == ["a1", "b1", "a2"]


def test_score_candidates_is_deterministic_and_combines_signals() -> None:
    retriever = object.__new__(KnowledgeRetriever)
    retriever.settings = SimpleNamespace(
        rag_vector_weight=0.25,
        rag_rerank_weight=0.30,
        rag_lexical_weight=0.45,
    )
    candidates = [
        make_candidate("relevant", "焦虑时可以通过呼吸和放松平复情绪。", score=0.7, rerank_score=0.7),
        make_candidate("generic", "这里介绍一般性的生活支持。", score=0.9, rerank_score=0.8),
    ]

    retriever._score_candidates("焦虑时可以做什么", candidates)

    assert candidates[0]["lexical_score"] > candidates[1]["lexical_score"]
    assert candidates[0]["combined_score"] > candidates[1]["combined_score"]


def test_select_rerank_candidates_uses_fast_scores() -> None:
    retriever = object.__new__(KnowledgeRetriever)
    retriever.settings = SimpleNamespace(
        rag_rerank_candidates=2,
        rag_vector_weight=0.25,
        rag_lexical_weight=0.45,
        rag_bm25_weight=0.20,
    )
    candidates = [
        make_candidate("generic", "一般性的生活支持。", score=0.95),
        make_candidate("relevant", "焦虑时可以通过呼吸和放松平复情绪。", score=0.70),
        make_candidate("other", "如何安排日常饮食和运动。", score=0.90),
    ]
    candidates[0]["bm25_normalized"] = 0.1
    candidates[1]["bm25_normalized"] = 1.0
    candidates[2]["bm25_normalized"] = 0.0

    selected = retriever._select_rerank_candidates("焦虑时怎么放松", candidates)

    assert len(selected) == 2
    assert selected[0]["chunk_id"] == "relevant"
    assert {item["chunk_id"] for item in selected} == {"relevant", "generic"}
