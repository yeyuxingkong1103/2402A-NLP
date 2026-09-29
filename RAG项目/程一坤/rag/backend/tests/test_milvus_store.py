from app.db.milvus_store import LegalMilvusStore


class FakeSchema:
    def __init__(self) -> None:
        self.fields: list[tuple[str, object, dict]] = []

    def add_field(self, name: str, data_type: object, **kwargs: object) -> None:
        self.fields.append((name, data_type, kwargs))


class FakeIndexParams:
    def __init__(self) -> None:
        self.indexes: list[tuple[str, dict]] = []

    def add_index(self, field_name: str, **kwargs: object) -> None:
        self.indexes.append((field_name, kwargs))


class FakeMilvusClient:
    def __init__(self) -> None:
        self.schema = FakeSchema()
        self.index_params = FakeIndexParams()
        self.created: dict | None = None
        self.upserted: tuple[str, list[dict]] | None = None
        self.loaded: str | None = None
        self.collections: set[str] = set()
        self.queried: tuple[str, str, list[str], int] | None = None
        self.query_result: list[dict] = []

    def has_collection(self, collection_name: str) -> bool:
        return collection_name in self.collections

    def create_schema(self, *, auto_id: bool) -> FakeSchema:
        assert auto_id is False
        return self.schema

    def prepare_index_params(self) -> FakeIndexParams:
        return self.index_params

    def create_collection(self, collection_name: str, *, schema: FakeSchema, index_params: FakeIndexParams) -> None:
        self.created = {
            "collection_name": collection_name,
            "schema": schema,
            "index_params": index_params,
        }
        self.collections.add(collection_name)

    def load_collection(self, collection_name: str) -> None:
        self.loaded = collection_name

    def upsert(self, collection_name: str, data: list[dict]) -> dict:
        self.upserted = (collection_name, data)
        return {"upsert_count": len(data), "ids": [row["chunk_key"] for row in data]}

    def query(self, collection_name: str, *, filter: str, output_fields: list[str], limit: int) -> list[dict]:
        self.queried = (collection_name, filter, output_fields, limit)
        return self.query_result


def test_legal_milvus_store_creates_collection_with_expected_schema() -> None:
    client = FakeMilvusClient()
    store = LegalMilvusStore(client=client, collection_name="legal_documents", dimension=1024)

    store.ensure_collection()

    assert client.created is not None
    assert client.created["collection_name"] == "legal_documents"
    assert client.loaded == "legal_documents"
    field_names = [field[0] for field in client.schema.fields]
    # 阶段 1 扩展了时效与法域字段；批次 37 追加 created_at/updated_at/summary（引用卡片摘要）
    assert field_names == [
        "chunk_key",
        "document_version_id",
        "chunk_type",
        "article_number",
        "retrieval_text",
        "law_name",
        "document_type",
        "jurisdiction",
        "authority_level",
        "effective_date",
        "expiration_date",
        "is_current",
        "article_path",
        "created_at",
        "updated_at",
        "summary",
        "dense_vector",
    ]
    # 批次 37：summary 可空 VARCHAR；created_at/updated_at 非空 INT64
    field_by_name = {field[0]: field for field in client.schema.fields}
    assert field_by_name["summary"][2].get("nullable") is True
    assert field_by_name["created_at"][2].get("nullable") is not True
    assert field_by_name["updated_at"][2].get("nullable") is not True
    vector_field = client.schema.fields[-1]
    assert vector_field[2]["dim"] == 1024
    assert client.index_params.indexes == [
        ("dense_vector", {"index_type": "AUTOINDEX", "metric_type": "COSINE"})
    ]


def test_legal_milvus_store_upserts_rows_with_dense_vector_field() -> None:
    client = FakeMilvusClient()
    store = LegalMilvusStore(client=client, collection_name="legal_documents", dimension=3)

    count = store.upsert(
        [
            {
                "chunk_key": "chunk-1",
                "document_version_id": 7,
                "chunk_type": "child",
                "article_number": "第一条",
                "retrieval_text": "测试文本",
                # 时效与法域字段（collection 已扩展）
                "law_name": "中华人民共和国劳动合同法",
                "document_type": "法律",
                "jurisdiction": "中国大陆",
                "authority_level": 2,
                "effective_date": 1372617600,
                "is_current": True,
                "article_path": "1",
                "vector": [0.1, 0.2, 0.3],
            }
        ]
    )

    assert count == 1
    payload = client.upserted[1][0]
    # 批次 37 新增键：时间戳（秒级，upsert 自动补）与 summary（未传时 None，可空）
    assert payload["created_at"] > 0
    assert payload["updated_at"] >= payload["created_at"]
    assert payload["summary"] is None
    assert client.upserted == (
        "legal_documents",
        [
            {
                "chunk_key": "chunk-1",
                "document_version_id": 7,
                "chunk_type": "child",
                "article_number": "第一条",
                "retrieval_text": "测试文本",
                "law_name": "中华人民共和国劳动合同法",
                "document_type": "法律",
                "jurisdiction": "中国大陆",
                "authority_level": 2,
                "effective_date": 1372617600,
                # 未传 expiration_date 时按 None 写入（字段可空）
                "expiration_date": None,
                "is_current": True,
                "article_path": "1",
                "created_at": payload["created_at"],
                "updated_at": payload["updated_at"],
                "summary": None,
                "dense_vector": [0.1, 0.2, 0.3],
            }
        ],
    )


def test_existing_chunk_keys_returns_only_keys_present_in_milvus() -> None:
    client = FakeMilvusClient()
    client.collections.add("legal_documents")
    client.query_result = [
        {"chunk_key": "chunk-1"},
        {"chunk_key": "chunk-3"},
    ]
    store = LegalMilvusStore(client=client, collection_name="legal_documents", dimension=3)

    result = store.existing_chunk_keys(["chunk-1", "chunk-2", "chunk-3", "chunk-4"])

    assert result == {"chunk-1", "chunk-3"}
    assert client.queried is not None
    collection_name, filter_expr, output_fields, limit = client.queried
    assert collection_name == "legal_documents"
    assert 'chunk_key in ["chunk-1","chunk-2","chunk-3","chunk-4"]' in filter_expr
    assert output_fields == ["chunk_key"]
    assert limit == 4


def test_existing_chunk_keys_returns_empty_set_when_candidate_list_is_empty() -> None:
    client = FakeMilvusClient()
    client.collections.add("legal_documents")
    store = LegalMilvusStore(client=client, collection_name="legal_documents", dimension=3)

    result = store.existing_chunk_keys([])

    assert result == set()
    assert client.queried is None
