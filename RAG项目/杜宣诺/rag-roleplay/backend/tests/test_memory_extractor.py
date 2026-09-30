import json

from app.core.fakes import FakeEmbedding
from app.services.memory_extractor import Memory, MemoryExtractor


class _NoopMilvus:
    async def hybrid_search(self, *a, **k):
        return []

    async def upsert_memories(self, memories):
        self.written = memories


class _ExistingMilvus:
    async def hybrid_search(self, *a, **k):
        # 返回一条已有记忆，dense 向量 [1.0]*1024
        return [{"dense_vector": [1.0] * 1024, "content": "已有记忆"}]

    async def upsert_memories(self, memories):
        self.written = memories


class _JsonLLM:
    async def chat(self, messages, **opts):
        return json.dumps({"memories": [
            {"type": "preference", "content": "喜欢猫", "importance": 7},
        ]})


async def test_extract_parses_json():
    ex = MemoryExtractor(_JsonLLM(), FakeEmbedding(), _NoopMilvus())
    mems = await ex.extract([{"role": "user", "content": "我喜欢猫"}])
    assert len(mems) == 1
    assert mems[0].content == "喜欢猫"


async def test_dedupe_filters_similar():
    ex = MemoryExtractor(_JsonLLM(), FakeEmbedding(), _ExistingMilvus())
    # FakeEmbedding: "y" 长度1 → [1.0]*1024，与已有 cosine=1.0 被剔除；
    # "abcdefg" 长度7 → 7%7=0 → [0.0]*1024，cosine=0 保留
    mems = [Memory(type="preference", content="y", importance=7),
            Memory(type="preference", content="abcdefg", importance=7)]
    kept = await ex.dedupe(mems, character_id=1, user_id=1)
    assert len(kept) == 1
    assert kept[0].content == "abcdefg"


async def test_run_writes_and_returns_count():
    milvus = _NoopMilvus()
    ex = MemoryExtractor(_JsonLLM(), FakeEmbedding(), milvus)
    n = await ex.run(session_id=1, user_id=1, character_id=7,
                     conversation=[{"role": "user", "content": "我喜欢猫"}])
    assert n == 1
    assert len(milvus.written) == 1
    assert milvus.written[0]["content"] == "喜欢猫"
