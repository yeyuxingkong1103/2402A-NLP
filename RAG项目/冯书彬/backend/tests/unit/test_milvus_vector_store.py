from backend.app.database.milvus import MilvusVectorStore
from backend.app.rag.result_merger import RetrievalFilters


class _FakeClient:
    def __init__(self):
        self.created = []
        self.rows = []
        self.search_filter = None

    def list_collections(self, timeout=None):
        return ["legal_material_chunks"]

    def has_collection(self, collection_name, timeout=None):
        return False

    def create_schema(self, **kwargs):
        class _Schema:
            def __init__(self):
                self.fields = []

            def add_field(self, *args, **kwargs):
                self.fields.append((args, kwargs))

        return _Schema()

    def prepare_index_params(self):
        class _IndexParams:
            def __init__(self):
                self.indexes = []

            def add_index(self, *args, **kwargs):
                self.indexes.append((args, kwargs))

        return _IndexParams()

    def create_collection(self, **kwargs):
        self.created.append(kwargs)

    def upsert(self, collection_name, data, timeout=None):
        self.rows.extend(data)
        return {"upsert_count": len(data)}

    def search(self, **kwargs):
        self.search_filter = kwargs.get("filter")
        return [[{"id": "row-1", "distance": 0.91, "entity": {"material_id": "m1", "version_id": "v1", "text": "正文", "metadata": {"article": "1", "source_url": "https://example.test/law", "publisher": "最高人民法院", "material_type": "judicial_interpretation", "status": "published", "searchable": True, "effective_from": "2021-01-01"}}}]]


def test_milvus_store_ensures_collection_and_writes_rows():
    store = MilvusVectorStore("http://localhost:19530", "legal_material_chunks")
    fake = _FakeClient()
    store._client = fake

    store.ensure_collection(1024)
    written = store.upsert([{"id": "row-1", "vector": [0.1]}])

    assert "schema" in fake.created[0]
    assert "index_params" in fake.created[0]
    assert written == 1


def test_milvus_search_maps_hits_and_filters_relationship_type():
    store = MilvusVectorStore("http://localhost:19530", "legal_material_chunks")
    fake = _FakeClient()
    store._client = fake

    rows = store.search([0.1, 0.2], top_k=3, filters=RetrievalFilters(relationship_type="divorce"))

    assert fake.search_filter == 'relationship_type == "divorce"'
    assert rows[0]["id"] == "row-1"
    assert rows[0]["material_id"] == "m1"
    assert rows[0]["version_id"] == "v1"
    assert rows[0]["text"] == "正文"
    assert rows[0]["score"] == 0.91
    assert rows[0]["metadata"]["article"] == "1"
    assert rows[0]["metadata"]["material"].id == "m1"
