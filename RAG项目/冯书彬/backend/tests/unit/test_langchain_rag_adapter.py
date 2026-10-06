import pytest
from threading import Event

from backend.app.langchain.prompt_adapter import build_langchain_llm_request
from backend.app.langchain.rag_adapter import LangChainUnavailableError, create_langchain_rag_decider
from backend.app.rag.pipeline import RetrievalDecision
from backend.app.rag.result_merger import RetrievedDocument
from backend.app.services.chat_service import _rag_shadow_summary, _schedule_langchain_shadow


class FakeReranker:
    def score(self, _query, documents):
        return [0.9 for _ in documents]


def test_langchain_adapter_requires_langchain_core(monkeypatch):
    real_import = __import__

    def blocked_import(name, *args, **kwargs):
        if name.startswith("langchain_core"):
            raise ImportError("blocked for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", blocked_import)
    with pytest.raises(LangChainUnavailableError):
        create_langchain_rag_decider(lambda _query: [], FakeReranker())


@pytest.mark.asyncio
async def test_langchain_adapter_preserves_retrieval_decision_contract():
    pytest.importorskip("langchain_core")

    async def candidates(_query):
        return []

    decider = create_langchain_rag_decider(candidates, FakeReranker())
    result = await decider("离婚财产怎么分？")

    assert isinstance(result, RetrievalDecision)
    assert result.can_answer is False
    assert result.reason == "no_candidates"


@pytest.mark.asyncio
async def test_langchain_retriever_maps_existing_documents_and_runs_reranker():
    pytest.importorskip("langchain_core")
    candidate = RetrievedDocument(
        id="doc-1",
        material_id="material-1",
        version_id="version-1",
        text="已发布法律依据",
        score=0.8,
        source="dense",
        metadata={"material": None},
    )

    async def candidates(_query):
        return [candidate]

    decider = create_langchain_rag_decider(candidates, FakeReranker())
    result = await decider("离婚财产怎么分？")

    assert isinstance(result, RetrievalDecision)
    assert result.reason == "insufficient_legal_basis"
    assert result.top_documents == []


def test_shadow_summary_contains_only_comparison_metadata():
    legacy = RetrievalDecision(can_answer=False, reason="insufficient_legal_basis", top_documents=[], citations=[])
    langchain = RetrievalDecision(can_answer=False, reason="no_candidates", top_documents=[], citations=[])

    summary = _rag_shadow_summary(legacy, langchain)

    assert summary["legacy_reason"] == "insufficient_legal_basis"
    assert summary["langchain_reason"] == "no_candidates"
    assert summary["comparison_status"] == "mismatch"
    assert "context" not in summary
    assert "text" not in summary


def test_shadow_summary_marks_matching_results():
    legacy = RetrievalDecision(can_answer=False, reason="no_candidates", top_documents=[], citations=[])
    langchain = RetrievalDecision(can_answer=False, reason="no_candidates", top_documents=[], citations=[])

    assert _rag_shadow_summary(legacy, langchain)["comparison_status"] == "match"


def test_shadow_scheduler_runs_after_request_loop_closes(monkeypatch):
    completed = Event()

    async def fake_shadow(_text, _legacy_result, _candidate_provider):
        completed.set()

    monkeypatch.setattr("backend.app.services.chat_service._run_langchain_shadow", fake_shadow)
    _schedule_langchain_shadow("脱敏测试问题", RetrievalDecision(False, "test", [], []), lambda _query: [])

    assert completed.wait(timeout=1)


def test_langchain_prompt_adapter_preserves_message_contract_and_limits_context():
    context = "A" * 1200 + "\n\n" + "B" * 1200 + "\n\n" + "C" * 1200
    decision = RetrievalDecision(can_answer=True, reason="ok", top_documents=[], citations=[], context=context)

    request = build_langchain_llm_request("离婚财产怎么分？", decision, "trace-langchain")

    assert request.messages[0] == {
        "role": "system",
        "content": "你是婚姻家事法律信息助手。只能依据给定资料回答，不能编造依据。",
    }
    assert request.messages[1]["role"] == "user"
    assert "用户问题：离婚财产怎么分？" in request.messages[1]["content"]
    assert len(request.messages[1]["content"].split("可用资料摘要：", 1)[1]) <= 2400
    assert request.trace_id == "trace-langchain"
    assert request.prompt_version == "chat-orchestration-langchain-v1"
