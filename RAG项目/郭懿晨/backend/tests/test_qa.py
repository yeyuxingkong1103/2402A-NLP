from backend.app.models import Citation, Chunk
from backend.app.qa import FALLBACK_ANSWER, build_answer_prompt, build_citations, build_fallback_response
from backend.app.vector_store import SearchResult


def test_build_fallback_response_has_no_citations():
    response = build_fallback_response()

    assert response.answer == FALLBACK_ANSWER
    assert response.citations == []
    assert response.fallback is True


def test_build_answer_prompt_contains_page_citation():
    citation = Citation(
        document_id="doc-1",
        file_name="a.pdf",
        page=5,
        category="正文",
        text="真实依据",
    )

    prompt = build_answer_prompt("问题", [citation])

    assert "只基于以下证据回答" in prompt
    assert "a.pdf 第 5 页" in prompt


def test_build_citations_keeps_search_score():
    result = SearchResult(
        chunk=Chunk(
            chunk_id="chunk-1",
            document_id="doc-1",
            page=5,
            category="正文",
            text="真实依据",
            source_span="page=5:block=1",
        ),
        score=0.8765,
    )

    citations = build_citations([result], {"doc-1": "a.pdf"})

    assert citations[0].score == 0.8765
    assert citations[0].file_name == "a.pdf"
