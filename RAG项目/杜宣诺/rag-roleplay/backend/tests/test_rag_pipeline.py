from app.core.fakes import FakeEmbedding, FakeRerank
from app.services.rag_pipeline import RAGPipeline, Source


class _StubMilvus:
    def __init__(self):
        self.calls = []

    async def hybrid_search(self, collection, query, filter_expr, top_k):
        self.calls.append((collection, filter_expr))
        if collection == "documents":
            return [
                {"id": 1, "text": "文档内容甲", "source": "甲.pdf", "doc_id": 10, "chunk_index": 0},
                {"id": 2, "text": "文档内容乙", "source": "乙.txt", "doc_id": 11, "chunk_index": 1},
            ]
        if collection == "character_settings":
            return [{"id": 3, "text": "角色设定片段", "setting_type": "persona", "character_id": 7}]
        if collection == "long_term_memory":
            return [{"id": 4, "content": "一段记忆", "memory_type": "fact", "user_id": 1, "character_id": 7}]
        return []


async def test_retrieve_hits_documents_globally():
    milvus = _StubMilvus()
    rag = RAGPipeline(FakeEmbedding(), FakeRerank(), milvus)
    r = await rag.retrieve(character_id=7, user_id=1, query="查询")
    assert len(r.documents) > 0
    # documents 分支应为全局过滤（id > 0），不绑定 character/user
    doc_filter = [f for c, f in milvus.calls if c == "documents"][0]
    assert "id > 0" in doc_filter
    assert "character_id" not in doc_filter


async def test_retrieve_preserves_source_metadata():
    milvus = _StubMilvus()
    rag = RAGPipeline(FakeEmbedding(), FakeRerank(), milvus)
    r = await rag.retrieve(character_id=7, user_id=1, query="查询")

    assert all(isinstance(s, Source) for s in r.documents + r.settings + r.memories)

    # documents：label=文档名，保留 doc_id / chunk_index
    assert r.documents[0].label in {"甲.pdf", "乙.txt"}
    assert r.documents[0].doc_id in {10, 11}
    assert r.documents[0].chunk_index in {0, 1}

    # settings：label=setting_type
    assert r.settings[0].label == "persona"
    assert r.settings[0].text == "角色设定片段"

    # memories：label=memory_type，text=content
    assert r.memories[0].label == "fact"
    assert r.memories[0].text == "一段记忆"


async def test_retrieve_degrades_when_milvus_fails():
    class _Boom:
        async def hybrid_search(self, *a, **k):
            raise RuntimeError("milvus down")
    rag = RAGPipeline(FakeEmbedding(), FakeRerank(), _Boom())
    r = await rag.retrieve(character_id=7, user_id=1, query="x")
    assert r.documents == [] and r.settings == [] and r.memories == []


async def test_source_to_dict_serializes():
    s = Source(type="document", text="片段", label="甲.pdf", doc_id=10, chunk_index=2)
    d = s.to_dict()
    assert d == {"type": "document", "label": "甲.pdf", "text": "片段", "doc_id": 10, "chunk_index": 2, "score": None}
