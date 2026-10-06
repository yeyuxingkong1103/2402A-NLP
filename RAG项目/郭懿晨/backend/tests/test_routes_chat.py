from pathlib import Path
from types import SimpleNamespace

import backend.app.routes_chat as routes_chat
from backend.app.models import Chunk
from backend.app.vector_store import SearchResult


class FakeSettings:
    qdrant_path = Path("data/qdrant")
    qdrant_collection = "rag_documents"
    bge_m3_model_path = Path("models/bge-m3")
    ollama_model = "deepseek-r1:7b"
    ollama_base_url = "http://localhost:11434"
    rerank_enabled = True
    reranker_model_path = Path("models/bge-reranker-v2-m3")
    reranker_device = None
    rerank_batch_size = 8
    retrieval_candidate_k = 3
    coarse_top_k = 3
    final_top_k = 3
    min_retrieval_score = 0.2


class FakeEmbedder:
    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path


class FakeVectorStore:
    def __init__(self, path: Path, collection_name: str) -> None:
        self.path = path
        self.collection_name = collection_name
        self.results: list[SearchResult] = []

    def search(self, query: str, embedder, limit: int):
        return list(self.results)


class FakeChain:
    def __init__(self, results: list[SearchResult]) -> None:
        self.results = results

    def retrieve(self, question: str) -> list[SearchResult]:
        return list(self.results)


class FakeStateStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def list_documents(self):
        return [SimpleNamespace(document_id="doc-1", file_name="a.pdf")]


class FakeQaService:
    def __init__(self, model: str, base_url: str) -> None:
        self.model = model
        self.base_url = base_url
        self.calls: list[list] = []

    def answer(self, question: str, citations):
        self.calls.append(list(citations))
        from backend.app.models import ChatResponse

        return ChatResponse(answer="ok", citations=list(citations), fallback=False)


def make_result(chunk_id: str, text: str, score: float) -> SearchResult:
    return SearchResult(
        chunk=Chunk(
            chunk_id=chunk_id,
            document_id="doc-1",
            page=7,
            category="正文",
            text=text,
            source_span="page=7:block=1",
        ),
        score=score,
    )


def test_chat_route_uses_retrieval_chain(monkeypatch):
    fake_service = FakeQaService("deepseek-r1:7b", "http://localhost:11434")
    chain = FakeChain([make_result("chunk-high", "高分片段", 0.42)])

    monkeypatch.setattr(routes_chat, "AppSettings", FakeSettings)
    monkeypatch.setattr(routes_chat, "BgeM3Embedder", FakeEmbedder)
    monkeypatch.setattr(routes_chat, "QdrantVectorStore", lambda path, collection: FakeVectorStore(path, collection))
    monkeypatch.setattr(routes_chat, "JsonStateStore", FakeStateStore)
    monkeypatch.setattr(routes_chat, "QaService", lambda model, base_url: fake_service)
    monkeypatch.setattr(routes_chat, "build_retrieval_chain", lambda settings, store, embedder: chain)

    response = routes_chat.chat(routes_chat.ChatRequest(question="怎么处理？"))

    assert response.answer == "ok"
    assert len(fake_service.calls) == 1
    assert len(fake_service.calls[0]) == 1
    assert fake_service.calls[0][0].score == 0.42
    assert response.citations[0].score == 0.42


def test_chat_route_returns_fallback_when_chain_empty(monkeypatch):
    fake_service = FakeQaService("deepseek-r1:7b", "http://localhost:11434")
    chain = FakeChain([])

    monkeypatch.setattr(routes_chat, "AppSettings", FakeSettings)
    monkeypatch.setattr(routes_chat, "BgeM3Embedder", FakeEmbedder)
    monkeypatch.setattr(routes_chat, "QdrantVectorStore", lambda path, collection: FakeVectorStore(path, collection))
    monkeypatch.setattr(routes_chat, "JsonStateStore", FakeStateStore)
    monkeypatch.setattr(routes_chat, "QaService", lambda model, base_url: fake_service)
    monkeypatch.setattr(routes_chat, "build_retrieval_chain", lambda settings, store, embedder: chain)

    response = routes_chat.chat(routes_chat.ChatRequest(question="怎么处理？"))

    assert response.fallback is True
    assert response.answer == "知识库中未找到相关依据。"
    assert fake_service.calls == []


def test_chat_route_returns_fallback_when_question_empty(monkeypatch):
    response = routes_chat.chat(routes_chat.ChatRequest(question="   "))

    assert response.fallback is True
    assert response.answer == "知识库中未找到相关依据。"
