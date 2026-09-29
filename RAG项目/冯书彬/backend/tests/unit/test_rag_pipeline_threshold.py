from datetime import datetime

from backend.app.models.knowledge_base import KnowledgeMaterial
from backend.app.rag.pipeline import decide_after_rerank
from backend.app.rag.result_merger import RetrievedDocument, RetrievalFilters


def test_low_reranker_score_blocks_answer():
    decision = decide_after_rerank(scores=[0.49], documents=[{"id": "doc-1"}], threshold=0.5)

    assert decision.can_answer is False
    assert decision.reason == "insufficient_legal_basis"
    assert decision.top_documents == []


def test_threshold_selects_top5_after_applicability_filtering():
    valid_material = _material("mat-valid", "published", True)
    draft_material = _material("mat-draft", "reviewed", True)
    filters = RetrievalFilters(query_date="2026-01-01", relationship_type="general", materials=[valid_material, draft_material])
    documents = [
        _document("valid-low", valid_material, 0.61),
        _document("draft-high", draft_material, 0.99),
    ]

    decision = decide_after_rerank(scores=[0.61, 0.99], documents=documents, filters=filters, threshold=0.5)

    assert decision.can_answer is True
    assert decision.reason == "ok"
    assert [doc.id for doc in decision.top_documents] == ["valid-low"]
    assert len(decision.citations) == 1

def test_missing_material_metadata_is_filtered_from_final_context():
    document = RetrievedDocument(
        id="no-material",
        material_id="mat-missing",
        version_id="mat-missing",
        text="第一条 缺少材料元数据。",
        score=0.99,
        source="dense",
        metadata={"article": "第一条"},
    )

    decision = decide_after_rerank(scores=[0.99], documents=[document], filters=None, threshold=0.5)

    assert decision.can_answer is False
    assert decision.reason == "insufficient_legal_basis"
    assert decision.top_documents == []
    assert decision.citations == []
    assert decision.context == ""


def test_invalid_high_score_does_not_unlock_valid_low_score():
    valid_material = _material("mat-valid-low", "published", True)
    draft_material = _material("mat-invalid-high", "reviewed", True)
    filters = RetrievalFilters(query_date="2026-01-01", relationship_type="general", materials=[valid_material, draft_material])
    documents = [
        _document("valid-low", valid_material, 0.49),
        _document("draft-high", draft_material, 0.99),
    ]

    decision = decide_after_rerank(scores=[0.49, 0.99], documents=documents, filters=filters, threshold=0.5)

    assert decision.can_answer is False
    assert decision.reason == "insufficient_legal_basis"
    assert decision.top_documents == []


def test_context_excludes_documents_without_citation():
    valid_material = _material("mat-traceable", "published", True)
    traceable = _document("traceable", valid_material, 0.91)
    untraceable = RetrievedDocument(
        id="untraceable",
        material_id="mat-untraceable",
        version_id="mat-untraceable",
        text="第二条 不能追溯引用。",
        score=0.9,
        source="dense",
        metadata={"article": "第二条"},
    )

    decision = decide_after_rerank(scores=[0.91, 0.9], documents=[traceable, untraceable], filters=None, threshold=0.5)

    assert decision.can_answer is True
    assert [doc.id for doc in decision.top_documents] == ["traceable"]
    assert len(decision.citations) == 1
    assert "mat-traceable" in decision.context
    assert "mat-untraceable" not in decision.context

def _material(material_id: str, status: str, searchable: bool) -> KnowledgeMaterial:
    return KnowledgeMaterial(
        id=material_id,
        snapshot_id=f"snap-{material_id}",
        source_url="https://example.test/law",
        publisher="全国人大",
        material_type="law",
        raw_text="第一条 为了测试检索依据。",
        attachments=[],
        status=status,
        searchable=searchable,
        created_at=datetime(2026, 1, 1),
        updated_at=datetime(2026, 1, 1),
        effective_from="2020-01-01",
    )


def _document(document_id: str, material: KnowledgeMaterial, score: float) -> RetrievedDocument:
    return RetrievedDocument(
        id=document_id,
        material_id=material.id,
        version_id=material.id,
        text="第一条 为了测试检索依据。",
        score=score,
        source="bm25",
        metadata={"material": material, "article": "第一条", "paragraph": None},
    )
