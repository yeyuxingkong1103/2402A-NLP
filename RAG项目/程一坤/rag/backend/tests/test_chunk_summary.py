"""批次 37 测试：chunk_summaries 表 / 索引带 summary / sources 组装 / Milvus schema 三新字段。

反向证明：先在旧实现（批次 36 前的备份）上跑，summary 相关用例应 import 失败或缺键变红；
再上实现跑绿。旧实现缺 ChunkSummary / build_citation_sources / schema 三字段，
本文件对"新增能力"的证明就是这些符号不存在 + 旧 payload 无 summary 键。
"""

from types import SimpleNamespace

from sqlalchemy import select

from app.chat.source_assembly import build_citation_sources
from app.db.milvus_store import SUMMARY_MAX_LENGTH, LegalMilvusStore
from app.db.sql_models import ChunkSummary, DocumentVersion
from app.db.vector_index_service import _fetch_summaries, index_awaiting_embeddings
from conftest import approve_pending_versions, build_validated_package, create_test_session


# ---------------- ChunkSummary 表 round-trip ----------------

def test_chunk_summary_roundtrip() -> None:
    session = create_test_session()
    session.add(ChunkSummary(chunk_key="k-1", summary="一句话摘要", model="test-model"))
    session.commit()

    row = session.scalars(
        select(ChunkSummary).where(ChunkSummary.chunk_key == "k-1")
    ).one()
    assert row.summary == "一句话摘要"
    assert row.model == "test-model"
    assert row.created_at is not None


# ---------------- 索引写入带 summary ----------------

def _prepare_indexed_session():
    session = create_test_session()
    import_package_ok = session  # 语义标注：import 后才有版本
    from app.db.import_service import import_package

    import_package(session, build_validated_package())
    approve_pending_versions(session)
    return session


def test_fetch_summaries_returns_only_existing_keys() -> None:
    session = _prepare_indexed_session()
    session.add(ChunkSummary(chunk_key="ver-a-parent", summary="父块摘要", model="m"))
    session.commit()

    result = _fetch_summaries(session, ["ver-a-parent", "ver-a-child", "missing"])
    assert result == {"ver-a-parent": "父块摘要"}


def test_index_rows_carry_summary_field() -> None:
    session = _prepare_indexed_session()
    session.add(ChunkSummary(chunk_key="ver-a-parent", summary="父块摘要", model="m"))
    session.commit()

    captured: list[dict] = []

    class FakeStore:
        def existing_chunk_keys(self, keys):
            return set()

        def upsert(self, rows):
            captured.extend(rows)

        def get_all_chunk_keys(self):
            return set()

        def delete_by_chunk_keys(self, keys):
            return 0

    class FakeEmbed:
        def embed(self, texts):
            return [[0.0, 0.0, 1.0] for _ in texts]

    result = index_awaiting_embeddings(
        session, embedding_client=FakeEmbed(), vector_store=FakeStore(), batch_size=10
    )
    assert result.status == "indexed"
    rows = {row["chunk_key"]: row for row in captured}
    # 有摘要的带摘要，没有的显式 None（Milvus 可空字段），不阻塞索引
    assert rows["ver-a-parent"]["summary"] == "父块摘要"
    assert rows["ver-a-child"]["summary"] is None


# ---------------- sources 组装 ----------------

def test_build_citation_sources_with_and_without_summary() -> None:
    # source_assembly 是纯函数（批次 37 修正：summary 由检索层挂在 articles 上，
    # 组装不碰数据库，单测零外部依赖）
    articles = [
        SimpleNamespace(
            chunk_key="k-1",
            document_title="测试法规",
            article_number="第一条",
            paragraph_number=None,
            summary="摘要-k-1",
        ),
        SimpleNamespace(
            chunk_key="k-2",
            document_title="测试法规",
            article_number="第二条",
            paragraph_number=None,
            summary=None,
        ),
    ]
    sources = build_citation_sources(articles)

    # 既有键保持不变 + 新增 summary（无摘要显式 None）
    assert sources[0] == {
        "chunk_id": "k-1",
        "law_name": "测试法规",
        "article_number": "第一条",
        "paragraph_number": None,
        "page": None,
        "summary": "摘要-k-1",
    }
    assert sources[1]["summary"] is None
    # 文档口径：无页码就是 None，不许用其他字段顶替
    assert all(source["page"] is None for source in sources)


# ---------------- Milvus schema 三新字段 ----------------

class _FakeField:
    def __init__(self, name, data_type, **kwargs):
        self.name = name
        self.data_type = data_type
        self.kwargs = kwargs


class _FakeSchema:
    def __init__(self):
        self.fields: list[_FakeField] = []

    def add_field(self, name, data_type, **kwargs):
        self.fields.append(_FakeField(name, data_type, **kwargs))


class _FakeIndexParams:
    def add_index(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


class _FakeMilvusClient:
    def __init__(self):
        self.loaded = False
        self.created = False

    def create_schema(self, auto_id=False):
        return _FakeSchema()

    def prepare_index_params(self):
        return _FakeIndexParams()

    def create_collection(self, name, schema, index_params):
        self.created = True

    def load_collection(self, name):
        self.loaded = True

    def has_collection(self, name):
        return False


def test_milvus_schema_contains_batch37_fields() -> None:
    store = LegalMilvusStore(
        client=_FakeMilvusClient(), collection_name="legal_documents", dimension=1024
    )
    schema = store._build_schema()
    names = [field.name for field in schema.fields]

    # 17 个字段 = 14 原有 + created_at/updated_at/summary
    assert len(names) == 17
    assert names[-2:] == ["summary", "dense_vector"] or "dense_vector" in names
    for required in ("created_at", "updated_at", "summary"):
        assert required in names, f"schema 缺批次 37 字段：{required}"
    summary_field = next(field for field in schema.fields if field.name == "summary")
    assert summary_field.kwargs.get("nullable") is True
    assert summary_field.kwargs.get("max_length") == SUMMARY_MAX_LENGTH
    # created_at/updated_at 非空 INT64
    for ts_field in ("created_at", "updated_at"):
        field = next(f for f in schema.fields if f.name == ts_field)
        assert field.kwargs.get("nullable") is not True


def test_upsert_payload_carries_timestamps_and_summary() -> None:
    store = LegalMilvusStore(
        client=_FakeMilvusClient(), collection_name="legal_documents", dimension=1024
    )
    captured: list[list[dict]] = []

    class _UpsertClient(_FakeMilvusClient):
        def upsert(self, name, payload):
            captured.append(payload)
            return {"upsert_count": len(payload)}

    store.client = _UpsertClient()
    row = {
        "chunk_key": "k-1",
        "document_version_id": 1,
        "chunk_type": "parent",
        "article_number": "第一条",
        "retrieval_text": "正文",
        "law_name": "测试法规",
        "document_type": "法律",
        "jurisdiction": "中国大陆",
        "authority_level": 1,
        "effective_date": 0,
        "expiration_date": None,
        "is_current": True,
        "article_path": "1",
        "summary": "一句话摘要",
        "vector": [0.0, 0.0, 1.0],
    }
    count = store.upsert([row])

    assert count == 1
    payload = captured[0][0]
    assert payload["summary"] == "一句话摘要"
    # created_at/updated_at 秒级时间戳，upsert 时自动补
    assert isinstance(payload["created_at"], int) and payload["created_at"] > 0
    assert payload["updated_at"] >= payload["created_at"]
    # row 未带 summary 时写 None（可空）
    row_without_summary = dict(row)
    row_without_summary.pop("summary")
    store.upsert([row_without_summary])
    assert captured[1][0]["summary"] is None


# ---------------- summarize_chunks：待处理集合口径 ----------------

def test_fetch_pending_chunks_parent_approved_and_resume() -> None:
    from app.cli.summarize_chunks import fetch_pending_chunks

    session = _prepare_indexed_session()
    # parent + child 各一块；只有 parent 应进入摘要集合
    pending = fetch_pending_chunks(session, force=False, limit=None)
    keys = {chunk.chunk_key for chunk in pending}
    assert keys == {"ver-a-parent"}
    assert all(chunk.chunk_type == "parent" for chunk in pending)

    # 断点续跑：已有摘要的 key 被跳过
    session.add(ChunkSummary(chunk_key="ver-a-parent", summary="已有", model="m"))
    session.commit()
    assert fetch_pending_chunks(session, force=False, limit=None) == []
    # force 时忽略已有摘要
    assert [c.chunk_key for c in fetch_pending_chunks(session, force=True, limit=None)] == [
        "ver-a-parent"
    ]

    # 未审核版本不进摘要集合
    session.rollback()
    versions = session.scalars(select(DocumentVersion)).all()
    for version in versions:
        version.version_status = "pending_review"
    session.commit()
    assert fetch_pending_chunks(session, force=True, limit=None) == []
