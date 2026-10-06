from sqlalchemy import select

from app.db.import_service import import_package
from app.db.sql_models import DocumentVersion
from app.db.vector_index_service import index_awaiting_embeddings
# 测试工具收敛到 conftest 一份（与 test_mysql_import_service 同一导入方式）
from conftest import approve_pending_versions, build_validated_package, create_test_session


class FakeEmbeddingClient:
    def __init__(self) -> None:
        self.texts: list[str] = []
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        self.texts.extend(texts)
        return [[float(index), 0.0, 1.0] for index, _ in enumerate(texts)]


class FakeVectorStore:
    def __init__(self) -> None:
        self.rows: list[dict] = []

    def existing_chunk_keys(self, chunk_keys: list[str]) -> set[str]:
        return {row["chunk_key"] for row in self.rows}.intersection(chunk_keys)

    def upsert(self, rows: list[dict]) -> None:
        self.rows.extend(rows)

    def existing_chunk_keys(self, chunk_keys: list[str]) -> set[str]:
        return {row["chunk_key"] for row in self.rows}.intersection(chunk_keys)


class FailingEmbeddingClient(FakeEmbeddingClient):
    def __init__(self) -> None:
        super().__init__()
        self.fail_on_call = 2

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self.fail_on_call == len(self.calls) + 1:
            self.calls.append(list(texts))
            self.fail_on_call = None
            raise TimeoutError("embedding timeout")
        return super().embed(texts)


def test_index_awaiting_embeddings_writes_vectors_and_marks_version_indexed() -> None:
    session = create_test_session()
    import_package(session, build_validated_package())
    approve_pending_versions(session)  # 模拟审核通过（索引只处理已发布版本）
    embedding_client = FakeEmbeddingClient()
    vector_store = FakeVectorStore()

    result = index_awaiting_embeddings(
        session,
        embedding_client=embedding_client,
        vector_store=vector_store,
        batch_size=10,
    )

    assert result.status == "indexed"
    assert result.indexed_versions == 1
    assert result.indexed_chunks == 2
    assert embedding_client.texts == [
        "测试法规 第一条 测试正文。",
        "测试法规 第一条 测试正文。",
    ]
    assert [row["chunk_key"] for row in vector_store.rows] == ["ver-a-parent", "ver-a-child"]
    assert vector_store.rows[0]["vector"] == [0.0, 0.0, 1.0]
    version_status = session.scalar(select(DocumentVersion.processing_status))
    assert version_status == "indexed"


def test_index_awaiting_embeddings_splits_embedding_requests_by_chunk_batch_size() -> None:
    session = create_test_session()
    import_package(session, build_validated_package())
    approve_pending_versions(session)  # 模拟审核通过（索引只处理已发布版本）
    embedding_client = FakeEmbeddingClient()
    vector_store = FakeVectorStore()

    index_awaiting_embeddings(
        session,
        embedding_client=embedding_client,
        vector_store=vector_store,
        batch_size=10,
        embedding_batch_size=1,
    )

    assert embedding_client.calls == [
        ["测试法规 第一条 测试正文。"],
        ["测试法规 第一条 测试正文。"],
    ]




def test_index_awaiting_embeddings_retries_only_chunks_not_written_before_failure() -> None:
    session = create_test_session()
    import_package(session, build_validated_package())
    approve_pending_versions(session)  # 模拟审核通过（索引只处理已发布版本）
    embedding_client = FailingEmbeddingClient()
    vector_store = FakeVectorStore()

    try:
        index_awaiting_embeddings(
            session,
            embedding_client=embedding_client,
            vector_store=vector_store,
            batch_size=10,
            embedding_batch_size=1,
        )
    except TimeoutError:
        pass
    else:
        raise AssertionError("expected embedding timeout")

    assert [row["chunk_key"] for row in vector_store.rows] == ["ver-a-parent"]
    assert session.scalar(select(DocumentVersion.processing_status)) == "awaiting_embedding"

    result = index_awaiting_embeddings(
        session,
        embedding_client=embedding_client,
        vector_store=vector_store,
        batch_size=10,
        embedding_batch_size=1,
    )

    assert result.status == "indexed"
    assert result.indexed_chunks == 1
    assert embedding_client.calls[-1] == ["测试法规 第一条 测试正文。"]
    assert [row["chunk_key"] for row in vector_store.rows] == [
        "ver-a-parent",
        "ver-a-child",
    ]
    session = create_test_session()
    embedding_client = FakeEmbeddingClient()
    vector_store = FakeVectorStore()

    result = index_awaiting_embeddings(
        session,
        embedding_client=embedding_client,
        vector_store=vector_store,
        batch_size=10,
    )

    assert result.status == "empty"
    assert result.indexed_versions == 0
    assert result.indexed_chunks == 0
    assert embedding_client.texts == []
    assert vector_store.rows == []
