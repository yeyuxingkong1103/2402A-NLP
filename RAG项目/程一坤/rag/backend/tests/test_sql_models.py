import pytest
from sqlalchemy import create_engine, inspect
from sqlalchemy.exc import IntegrityError

from app.db.base import Base
from app.db.sql_models import (
    CrawlRecord,
    Document,
    DocumentChunk,
    DocumentVersion,
    ImportRecord,
    User,
)
# 测试工具收敛到 conftest 一份（与 test_law_models 同一导入方式）
from tests.conftest import create_test_session


def test_metadata_creates_required_tables() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")

    Base.metadata.create_all(engine)

    table_names = set(inspect(engine).get_table_names())
    assert {
        "documents",
        "document_versions",
        "document_chunks",
        "crawl_records",
        "import_records",
        "users",
    }.issubset(table_names)


@pytest.mark.parametrize(
    ("first", "second"),
    [
        (
            Document(
                document_key="doc-1",
                source_url="https://example.com/a",
                title="文档 A",
            ),
            Document(
                document_key="doc-1",
                source_url="https://example.com/b",
                title="文档 B",
            ),
        ),
        (
            Document(
                document_key="doc-1",
                source_url="https://example.com/a",
                title="文档 A",
            ),
            Document(
                document_key="doc-2",
                source_url="https://example.com/a",
                title="文档 B",
            ),
        ),
        # 认证落库后 users 表唯一业务键为 email（批次5 移除 username 列）
        (
            User(user_key="a" * 32, email="alice@qq.com", password_hash="hash-1", is_active=True),
            User(user_key="b" * 32, email="alice@qq.com", password_hash="hash-2", is_active=True),
        ),
    ],
)
def test_unique_top_level_business_keys_are_enforced(first, second) -> None:
    session = create_test_session()
    session.add(first)
    session.commit()

    session.add(second)
    with pytest.raises(IntegrityError):
        session.commit()


def test_unique_version_chunk_and_import_keys_are_enforced() -> None:
    session = create_test_session()
    document = Document(
        document_key="doc-1",
        source_url="https://example.com/a",
        title="文档 A",
    )
    session.add(document)
    session.flush()

    version = DocumentVersion(
        version_key="ver-1",
        document_id=document.id,
        content_hash="hash-a",
        raw_file_path="raw/a.html",
        media_type="text/html",
        cleaned_content="第一条 测试正文。",
        version_status="new",
        processing_status="awaiting_embedding",
    )
    session.add(version)
    session.flush()

    # 这些记录代表后续入库和关键词检索依赖的最小约束面。
    session.add_all(
        [
            DocumentChunk(
                chunk_key="chunk-1",
                document_version_id=version.id,
                parent_chunk_key=None,
                chunk_type="parent",
                article_number="第一条",
                sequence=1,
                content="第一条 测试正文。",
                retrieval_text="文档 A 第一条 测试正文。",
            ),
            CrawlRecord(
                package_id="pkg-1",
                source_url="https://example.com/a",
                document_id=document.id,
                document_version_id=version.id,
                status="success",
                http_status=200,
                content_hash="hash-a",
                raw_file_path="raw/a.html",
            ),
            ImportRecord(
                package_id="pkg-1",
                schema_version="1.0.0",
                pipeline_version="0.1.0",
                status="imported",
                document_version_id=version.id,
            ),
        ]
    )
    session.commit()

    duplicate_version = DocumentVersion(
        version_key="ver-1",
        document_id=document.id,
        content_hash="hash-b",
        raw_file_path="raw/b.html",
        media_type="text/html",
        cleaned_content="第二条 测试正文。",
        version_status="updated",
        processing_status="awaiting_embedding",
    )
    session.add(duplicate_version)
    with pytest.raises(IntegrityError):
        session.commit()

    session.rollback()
    session.add(
        DocumentChunk(
            chunk_key="chunk-1",
            document_version_id=version.id,
            parent_chunk_key=None,
            chunk_type="parent",
            article_number="第二条",
            sequence=2,
            content="第二条 测试正文。",
            retrieval_text="文档 A 第二条 测试正文。",
        )
    )
    with pytest.raises(IntegrityError):
        session.commit()

    session.rollback()
    session.add(
        ImportRecord(
            package_id="pkg-1",
            schema_version="1.0.0",
            pipeline_version="0.1.0",
            status="already_imported",
            document_version_id=version.id,
        )
    )
    with pytest.raises(IntegrityError):
        session.commit()
