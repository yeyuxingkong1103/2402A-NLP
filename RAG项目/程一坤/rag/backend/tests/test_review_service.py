"""阶段6 审核与发布：能力层测试（review_service）。

核心语义：
- 新导入版本一律 pending_review，未审核内容不进 Milvus（索引侧只扫 approved）
- approve = 置状态 + 留痕 + 切生效指针 + 触发向量索引（索引校验通过才算成功）
- reject = 置状态 + 留痕 + 删除该版本已存在的向量（MySQL 内容保留）
- 列表按提交时间倒序 + 分页
"""
from datetime import datetime

from sqlalchemy import select

from app.db.import_service import import_package
from app.db.sql_models import Document, DocumentChunk, DocumentVersion
from app.review.review_service import list_review_documents, review_document
from sqlalchemy.orm import sessionmaker

from conftest import approve_pending_versions, build_validated_package, create_test_session


class FakeEmbeddingClient:
    def __init__(self) -> None:
        self.texts: list[str] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.texts.extend(texts)
        return [[float(i), 0.0, 1.0] for i, _ in enumerate(texts)]


class FakeVectorStore:
    """记录 upsert/delete 的向量库替身。"""

    def __init__(self) -> None:
        self.rows: list[dict] = []
        self.deleted: list[str] = []

    def existing_chunk_keys(self, chunk_keys) -> set[str]:
        return {row["chunk_key"] for row in self.rows}.intersection(chunk_keys)

    def upsert(self, rows: list[dict]) -> None:
        self.rows.extend(rows)

    def get_all_chunk_keys(self) -> set[str]:
        return {row["chunk_key"] for row in self.rows}

    def delete_by_chunk_keys(self, chunk_keys) -> int:
        keys = set(chunk_keys)
        before = len(self.rows)
        self.rows = [row for row in self.rows if row["chunk_key"] not in keys]
        self.deleted.extend(keys)
        return before - len(self.rows)


def make_store_and_embedding():
    return FakeVectorStore(), FakeEmbeddingClient()


# ==================== 导入侧：新版本即待审核 ====================


def test_import_creates_pending_review_version() -> None:
    """新导入版本的 version_status 必须是 pending_review（未审核不可发布）。"""
    session = create_test_session()
    import_package(session, build_validated_package())

    version = session.scalar(select(DocumentVersion))
    assert version.version_status == "pending_review"


# ==================== approve：发布 + 索引 + 留痕 ====================


def test_approve_publishes_indexes_and_records_reviewer() -> None:
    session = create_test_session()
    result = import_package(session, build_validated_package())
    document = session.scalar(select(Document))
    store, embedding = make_store_and_embedding()

    summary = review_document(
        session,
        document_id=document.id,
        reviewer_user_key="a" * 32,
        decision="approve",
        review_note="已核对官方来源",
        vector_store=store,
        embedding_client=embedding,
    )

    version = session.scalar(select(DocumentVersion))
    assert version.version_status == "approved"
    assert version.processing_status == "indexed"
    assert version.reviewed_by == "a" * 32
    assert version.review_note == "已核对官方来源"
    assert version.reviewed_at is not None
    # 生效指针切到被批准的版本
    session.refresh(document)
    assert document.current_version_id == version.id
    # 向量已写入
    assert {row["chunk_key"] for row in store.rows} == {
        "ver-a-parent", "ver-a-child",
    }
    assert summary["version_status"] == "approved"


def test_approve_fails_when_index_verification_fails() -> None:
    """approve 后向量校验发现缺向量 → 抛错，不得静默当作发布成功。"""
    session = create_test_session()
    import_package(session, build_validated_package())
    document = session.scalar(select(Document))
    store, embedding = make_store_and_embedding()

    class HalfFailingStore(FakeVectorStore):
        """upsert 时悄悄丢掉一个向量，模拟索引写坏。"""

        def upsert(self, rows: list[dict]) -> None:
            super().upsert(rows[:-1])

    summary = review_document(
        session,
        document_id=document.id,
        reviewer_user_key="a" * 32,
        decision="approve",
        review_note=None,
        vector_store=HalfFailingStore(),
        embedding_client=embedding,
    )
    assert summary["index_verified"] is False
    # 状态已置 approved（人工决策已发生），但处理状态停在被标记位，
    # 供后续重试索引；不能谎报 indexed
    version = session.scalar(select(DocumentVersion))
    assert version.processing_status != "indexed"


# ==================== reject：驳回 + 删向量 + 内容保留 ====================


def test_reject_removes_vectors_keeps_mysql_content() -> None:
    session = create_test_session()
    import_package(session, build_validated_package())
    document = session.scalar(select(Document))
    store, embedding = make_store_and_embedding()

    # 模拟历史遗留：该版本的向量已经在库里（老机制下导入即索引）
    chunks = session.scalars(select(DocumentChunk)).all()
    store.upsert(
        [
            {
                "chunk_key": chunk.chunk_key,
                "vector": [0.0, 0.0, 1.0],
            }
            for chunk in chunks
        ]
    )
    assert len(store.rows) == 2

    summary = review_document(
        session,
        document_id=document.id,
        reviewer_user_key="b" * 32,
        decision="reject",
        review_note="来源无法核实",
        vector_store=store,
        embedding_client=embedding,
    )

    version = session.scalar(select(DocumentVersion))
    assert version.version_status == "rejected"
    assert version.reviewed_by == "b" * 32
    assert version.review_note == "来源无法核实"
    # Milvus 里没有它的向量了
    version_keys = {chunk.chunk_key for chunk in chunks}
    assert version_keys & store.get_all_chunk_keys() == set()
    assert summary["deleted_vectors"] == 2
    # MySQL 正文保留（内容不被删除）
    remaining = session.scalars(select(DocumentChunk)).all()
    assert len(remaining) == 2


# ==================== 边界 ====================


def test_review_without_pending_version_is_rejected() -> None:
    """已审核过的文档再次 review → 明确报错（无待审核版本）。"""
    session = create_test_session()
    import_package(session, build_validated_package())
    document = session.scalar(select(Document))
    store, embedding = make_store_and_embedding()
    review_document(
        session,
        document_id=document.id,
        reviewer_user_key="a" * 32,
        decision="approve",
        review_note=None,
        vector_store=store,
        embedding_client=embedding,
    )

    try:
        review_document(
            session,
            document_id=document.id,
            reviewer_user_key="a" * 32,
            decision="approve",
            review_note=None,
            vector_store=store,
            embedding_client=embedding,
        )
        raise AssertionError("应当抛 ValueError")
    except ValueError as error:
        assert "待审核" in str(error)


def test_invalid_decision_rejected() -> None:
    session = create_test_session()
    import_package(session, build_validated_package())
    document = session.scalar(select(Document))
    store, embedding = make_store_and_embedding()

    try:
        review_document(
            session,
            document_id=document.id,
            reviewer_user_key="a" * 32,
            decision="delete",
            review_note=None,
            vector_store=store,
            embedding_client=embedding,
        )
        raise AssertionError("应当抛 ValueError")
    except ValueError as error:
        assert "decision" in str(error)


def test_multi_version_coexist_approve_switches_pointer_only() -> None:
    """同文档新版本导入后：旧 approved 版本行与内容原样保留，指针只切换。"""
    session = create_test_session()
    import_package(session, build_validated_package())
    document = session.scalar(select(Document))
    store, embedding = make_store_and_embedding()
    review_document(
        session,
        document_id=document.id,
        reviewer_user_key="a" * 32,
        decision="approve",
        review_note=None,
        vector_store=store,
        embedding_client=embedding,
    )
    old_version = session.scalar(select(DocumentVersion))
    old_content = old_version.cleaned_content

    # 内容变化 → 新版本（pending_review）
    import_package(
        session,
        build_validated_package(
            package_id="pkg-2",
            content_hash="hash-b",
            version_key="ver-b",
        ),
    )
    versions = session.scalars(select(DocumentVersion).order_by(DocumentVersion.id)).all()
    assert len(versions) == 2
    assert versions[0].version_status == "approved"  # 旧版本未被改写
    assert versions[1].version_status == "pending_review"
    assert versions[0].cleaned_content == old_content

    review_document(
        session,
        document_id=document.id,
        reviewer_user_key="a" * 32,
        decision="approve",
        review_note=None,
        vector_store=store,
        embedding_client=embedding,
    )
    session.refresh(document)
    session.refresh(versions[1])
    assert document.current_version_id == versions[1].id
    assert versions[1].version_status == "approved"


# ==================== 列表：过滤 + 分页 + 倒序 ====================


def test_list_review_documents_filters_and_paginates() -> None:
    session = create_test_session()
    # 三个文档各一个 pending 版本
    for index, content_hash in enumerate(["h-1", "h-2", "h-3"], start=1):
        import_package(
            session,
            build_validated_package(
                package_id=f"pkg-{index}",
                source_url=f"https://example.com/law/{index}",
                document_key=f"doc-{index}",
                version_key=f"ver-{index}",
                content_hash=content_hash,
            ),
        )
    document = session.scalar(select(Document))
    store, embedding = make_store_and_embedding()
    # 批准其中一个
    review_document(
        session,
        document_id=document.id,
        reviewer_user_key="a" * 32,
        decision="approve",
        review_note=None,
        vector_store=store,
        embedding_client=embedding,
    )

    pending_page1 = list_review_documents(
        session, status="pending_review", page=1, page_size=2
    )
    assert pending_page1["total"] == 2
    assert len(pending_page1["items"]) == 2
    pending_page2 = list_review_documents(
        session, status="pending_review", page=2, page_size=2
    )
    assert pending_page2["total"] == 2
    assert pending_page2["items"] == []
    # 按提交时间倒序：page1 第一条是最后导入的文档
    assert pending_page1["items"][0]["version_key"] == "ver-3"

    approved = list_review_documents(
        session, status="approved", page=1, page_size=10
    )
    assert approved["total"] == 1
    assert approved["items"][0]["reviewed_by"] == "a" * 32


# ==================== 6.3② 检索侧兜底 ====================


def test_vector_fetch_chunks_filters_unapproved_versions() -> None:
    """历史遗留向量召回的未审核 chunk，取正文时被兜底过滤。"""
    from app.retrieval.vector_search import LegalRetriever

    session = create_test_session()
    import_package(session, build_validated_package())
    chunks = session.scalars(select(DocumentChunk)).all()
    keys = [chunk.chunk_key for chunk in chunks]

    # 未审核（pending_review）：兜底过滤后取不到正文
    fetched = LegalRetriever._fetch_chunks(session, keys)
    assert fetched == []

    # 审核通过后：同一批 chunk 能取到正文
    approve_pending_versions(session)
    fetched = LegalRetriever._fetch_chunks(session, keys)
    assert {row["chunk_key"] for row in fetched} == set(keys)


def test_keyword_index_excludes_unapproved_versions() -> None:
    """BM25 索引只收录 approved 版本；未审核内容不进关键词索引。"""
    import pytest

    from app.retrieval.keyword_search import KeywordSearchError, KeywordSearcher

    session = create_test_session()
    import_package(session, build_validated_package())
    factory = sessionmaker(bind=session.get_bind(), expire_on_commit=False)

    searcher = KeywordSearcher(session_factory=factory)
    # 未审核：索引里什么都没有（空库按既有语义抛 KeywordSearchError）
    with pytest.raises(KeywordSearchError):
        searcher.refresh()

    approve_pending_versions(session)
    assert searcher.refresh() == 2  # 审核通过：父块+子块都进索引
