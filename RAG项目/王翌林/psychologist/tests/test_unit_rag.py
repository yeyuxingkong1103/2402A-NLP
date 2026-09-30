"""单元测试：RAG 组件——embedder/reranker 单例契约、retriever 编排、pipeline 组合根。

真实模型加载与 Milvus 连通属 integration 用例（见 test_integration_models.py /
test_integration_milvus.py），本文件全部使用替身，不加载模型、不触网。
"""
import pytest

from src.rag import pipeline
from src.rag.embedder import Embedder, get_embedder
from src.rag.reranker import Reranker, get_reranker
from src.rag.retriever import retrieve
from src.services import retrieval_service

DIM = 1024


class FakeEmbedder:
    dim = DIM

    def encode_query(self, text):
        return [0.1] * DIM

    def encode(self, texts, **kwargs):
        return [[0.1] * DIM for _ in texts]


class FakeReranker:
    def rerank(self, query, candidates, top_n=None, text_field="text"):
        scored = [dict(c, rerank_score=0.9 if "认知" in c.get(text_field, "") else 0.05)
                  for c in candidates]
        scored.sort(key=lambda x: x["rerank_score"], reverse=True)
        return scored[: top_n or len(scored)]


class FakeMilvus:
    def __init__(self):
        self.calls = []

    def hybrid_search_knowledge(self, persona_id, dense, text, top_k=None):
        self.calls.append(("hybrid", persona_id))
        return [{"doc_id": 1, "text": "认知行为疗法……", "score": 0.8},
                {"doc_id": 2, "text": "无关内容", "score": 0.4}]

    def dense_search_knowledge(self, persona_id, dense, top_k=None):
        self.calls.append(("dense", persona_id))
        return [{"doc_id": 9, "text": "降级检索结果", "score": 0.5}]


# ---------------- embedder / reranker 单例契约（不加载模型） ----------------
def test_embedder_singleton():
    assert get_embedder() is get_embedder()


def test_embedder_lazy_and_dim():
    e = Embedder()
    assert e.loaded is False          # 未触发 load，不应加载真实模型
    assert e.dim == 1024


def test_reranker_singleton():
    assert get_reranker() is get_reranker()


def test_reranker_lazy():
    r = Reranker()
    assert r.loaded is False
    assert "bge-reranker" in r.model_path


# ---------------- retriever 检索编排 ----------------
@pytest.fixture
def fake_rag_stack(monkeypatch):
    milvus = FakeMilvus()
    monkeypatch.setattr("src.rag.retriever.get_embedder", lambda: FakeEmbedder())
    monkeypatch.setattr("src.rag.retriever.get_reranker", lambda: FakeReranker())
    monkeypatch.setattr("src.rag.retriever.milvus_db", milvus)
    return milvus


def test_retrieve_hybrid_pipeline(fake_rag_stack):
    hits = retrieve(persona_id=1, query="我总是灾难化思考怎么办")
    assert fake_rag_stack.calls[0][0] == "hybrid"
    assert hits and hits[0]["doc_id"] == 1          # 相关片段排前
    assert all("rerank_score" in h for h in hits)


def test_retrieve_falls_back_to_dense(fake_rag_stack):
    fake_rag_stack.hybrid_search_knowledge = lambda *a, **k: []   # 混合无结果
    hits = retrieve(persona_id=1, query="test")
    assert any(call[0] == "dense" for call in fake_rag_stack.calls)
    assert hits[0]["doc_id"] == 9


def test_retrieve_empty_candidates(fake_rag_stack):
    fake_rag_stack.hybrid_search_knowledge = lambda *a, **k: []
    fake_rag_stack.dense_search_knowledge = lambda *a, **k: []
    assert retrieve(persona_id=1, query="test") == []


def test_rewrite_disabled_returns_original(monkeypatch):
    from src.core.config import settings
    monkeypatch.setattr(settings, "query_rewrite_enabled", False)
    assert retrieval_service.rewrite_query("原始问题", []) == "原始问题"


def test_rewrite_merges_when_stub_empty(monkeypatch):
    """simple_complete 返回空串（测试环境）→ 改写回退为原问题。"""
    from src.core.config import settings
    monkeypatch.setattr(settings, "query_rewrite_enabled", True)
    assert retrieval_service.rewrite_query("原始问题", []) == "原始问题"


def test_search_with_query_rewrite_returns_pair(fake_rag_stack, monkeypatch):
    from src.core.config import settings
    monkeypatch.setattr(settings, "query_rewrite_enabled", False)
    hits, rewritten = retrieval_service.search_with_query_rewrite(1, "问题", [])
    assert rewritten == "问题" and isinstance(hits, list)


# ---------------- pipeline 组合根 ----------------
def test_pipeline_offline_delegates(monkeypatch):
    seen = {}

    def fake_ingest(db, file_path, persona_id, strategy="paragraph", title=None):
        seen.update(path=file_path, persona=persona_id, strategy=strategy)
        return {"doc_id": 7}

    monkeypatch.setattr("src.services.knowledge_service.ingest_file", fake_ingest)
    result = pipeline.offline_ingest_file(db=object(), file_path="/tmp/a.txt",
                                          persona_id=3, strategy="sentence")
    assert result == {"doc_id": 7}
    assert seen == {"path": "/tmp/a.txt", "persona": 3, "strategy": "sentence"}


def test_pipeline_online_delegates(monkeypatch):
    monkeypatch.setattr("src.services.rag_service.answer",
                        lambda db, uid, pid, msg, cid=None: {"user": uid, "persona": pid})
    result = pipeline.online_answer(db=object(), user_id=5, persona_id=6, message="hi")
    assert result == {"user": 5, "persona": 6}


def test_pipeline_stream_delegates(monkeypatch):
    def fake_stream(db, uid, pid, msg, cid=None):
        yield "delta"

    monkeypatch.setattr("src.services.rag_service.answer_stream", fake_stream)
    assert list(pipeline.online_answer_stream(object(), 1, 2, "m")) == ["delta"]
