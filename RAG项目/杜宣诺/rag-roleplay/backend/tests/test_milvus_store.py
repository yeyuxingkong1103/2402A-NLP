import pytest

from app.core.fakes import FakeEmbedding
from app.store.milvus_store import MilvusStore


class _FakeSchema:
    def __init__(self):
        self.fields = []

    def add_field(self, name, dtype, **kwargs):
        self.fields.append((name, dtype))


class _FakeIndexParams:
    def add_index(self, **k):
        pass


class FakeMilvusClient:
    def __init__(self, *a, **k):
        self.collections = {}
        self.indexes = {}
        self.loaded = set()
        self.rows = {"character_settings": [], "long_term_memory": [], "documents": []}
        self._id = 0

    @staticmethod
    def create_schema(auto_id=True, enable_dynamic_field=True):
        return _FakeSchema()

    def has_collection(self, name):
        return name in self.collections

    def prepare_index_params(self):
        return _FakeIndexParams()

    def create_collection(self, name, **k):
        self.collections[name] = True

    def list_indexes(self, name):
        return self.indexes.get(name, [])

    def create_index(self, name, idx):
        self.indexes[name] = [name]

    def load_collection(self, name):
        self.loaded.add(name)

    def insert(self, name, rows):
        for r in rows:
            self._id += 1
            self.rows[name].append({**r, "id": self._id})

    def search(self, name, data, anns_field, limit, filter, output_fields):
        # 返回全部已存行（模拟命中）
        return [[{"id": r["id"], "entity": r} for r in self.rows[name][:limit]]]

    def delete(self, name, ids=None, filter=None):
        self.rows[name] = []

    def close(self):
        pass


@pytest.fixture
async def store(monkeypatch):
    from app.store import milvus_store as ms
    monkeypatch.setattr(ms, "MilvusClient", FakeMilvusClient)
    s = MilvusStore(FakeEmbedding())
    await s.init_collections()
    return s


async def test_upsert_and_hybrid_search(store):
    await store.upsert_settings(7, [{"setting_type": "persona", "text": "温柔的图书馆管理员"}])
    res = await store.hybrid_search("character_settings", "她是什么性格", "character_id == 7", 5)
    assert len(res) == 1
    assert res[0]["character_id"] == 7


async def test_upsert_memories(store):
    await store.upsert_memories([{"user_id": 1, "character_id": 7, "memory_type": "preference", "content": "喜欢猫", "importance": 7}])
    assert len(store.client.rows["long_term_memory"]) == 1


async def test_upsert_documents(store):
    await store.upsert_documents([
        {"doc_id": 1, "chunk_index": 0, "source": "a.txt", "text": "第一块", "created_at": 1},
        {"doc_id": 1, "chunk_index": 1, "source": "a.txt", "text": "第二块", "created_at": 1},
    ])
    rows = store.client.rows["documents"]
    assert len(rows) == 2
    assert rows[0]["doc_id"] == 1 and rows[0]["chunk_index"] == 0
    assert "dense_vector" in rows[0] and "sparse_vector" in rows[0]
