"""测试增量导入逻辑：new / unchanged / updated 三种状态以及旧版本保留。

任务书 2-B 要求：
- 按 source_url + content_hash 判定状态
- unchanged：跳过，不创建重复版本
- new：首次导入
- updated：创建新版本，旧版本保留为历史可查（replace_existing=False）
- replace_existing=True：强制替换，删除旧版本
"""
import hashlib
from pathlib import Path
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker
from app.db.sql_models import Base, Document, DocumentVersion, DocumentChunk
from app.db.import_service import import_package
from app.pipeline.package_validator import ValidatedPackage, PackageManifest
from app.pipeline.package_models import PackageData, DocumentEntry, VersionEntry, ChunkEntry, CrawlEntry


def _create_minimal_package(
    document_id: str,
    source_url: str,
    content: str,
    version_id: str = "v1",
    title: str | None = None,
) -> ValidatedPackage:
    """构造测试用 ValidatedPackage（最小结构）。"""
    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

    # 构造 PackageData，使用正确的 dataclass 类型
    data = PackageData(
        document=DocumentEntry(
            document_id=document_id,
            source_url=source_url,
            title=title or "未命名文档",
        ),
        version=VersionEntry(
            version_id=version_id,
            document_id=document_id,
            content_hash=content_hash,
            media_type="text/html",
            cleaned_content=content,
            raw_file_path="test.html",
        ),
        chunks=(
            ChunkEntry(
                chunk_id=f"{version_id}_chunk1",
                version_id=version_id,
                parent_chunk_id=None,
                chunk_type="article",
                article_number=None,
                sequence=0,
                content=content[:100] if len(content) > 100 else content,
                retrieval_text=content[:100] if len(content) > 100 else content,
            ),
        ),
        crawl=CrawlEntry(
            crawl_id=f"crawl_{document_id}_{version_id}",
            source_id=document_id,
            source_url=source_url,
            http_status=200,
            content_hash=content_hash,
            raw_file_path="test.html",
            collected_at="2024-01-01T00:00:00Z",
        ),
    )

    # 构造 PackageManifest
    manifest = PackageManifest(
        schema_version="1.0.0",
        pipeline_version="test",
        package_id=f"test_{document_id}_{version_id}",
        created_at="2024-01-01T00:00:00Z",
        source_id=document_id,
        source_url=source_url,
        record_counts={"documents": 1, "versions": 1, "chunks": 1, "crawls": 1},
        file_hashes={},
    )

    return ValidatedPackage(manifest=manifest, data=data)


def test_incremental_new():
    """测试状态：new（首次导入）"""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)

    package = _create_minimal_package(
        "doc1", "https://example.com/doc1", "第一条 内容A", "v1", "文档1"
    )

    with SessionLocal() as session:
        result = import_package(session, package, replace_existing=False)
        session.commit()

    assert result.status == "imported"
    assert result.version_status == "new"

    # 验证数据库状态
    with SessionLocal() as session:
        doc = session.scalar(select(Document).where(Document.source_url == "https://example.com/doc1"))
        assert doc is not None
        assert doc.title == "文档1"
        assert doc.current_version_id is not None

        versions = session.scalars(select(DocumentVersion).where(DocumentVersion.document_id == doc.id)).all()
        assert len(versions) == 1
        assert versions[0].version_key == "v1"


def test_incremental_unchanged():
    """测试状态：unchanged（content_hash 相同，不创建重复版本）"""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)

    content = "第一条 内容A"
    package1 = _create_minimal_package("doc1", "https://example.com/doc1", content, "v1")
    package2 = _create_minimal_package("doc1", "https://example.com/doc1", content, "v1_retry")

    # 第一次导入
    with SessionLocal() as session:
        result1 = import_package(session, package1)
        session.commit()
    assert result1.status == "imported"
    assert result1.version_status == "new"

    # 第二次导入相同内容
    with SessionLocal() as session:
        result2 = import_package(session, package2)
        session.commit()
    assert result2.status == "already_imported"
    assert result2.version_status == "unchanged"

    # 验证只有一个版本
    with SessionLocal() as session:
        doc = session.scalar(select(Document).where(Document.source_url == "https://example.com/doc1"))
        versions = session.scalars(select(DocumentVersion).where(DocumentVersion.document_id == doc.id)).all()
        assert len(versions) == 1


def test_incremental_updated_preserves_history():
    """测试状态：updated（content_hash 不同，创建新版本，旧版本保留）"""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)

    content_v1 = "第一条 内容A"
    content_v2 = "第一条 内容B（修订）"

    package1 = _create_minimal_package("doc1", "https://example.com/doc1", content_v1, "v1")
    package2 = _create_minimal_package("doc1", "https://example.com/doc1", content_v2, "v2")

    # 第一次导入
    with SessionLocal() as session:
        result1 = import_package(session, package1)
        session.commit()
    assert result1.version_status == "new"

    with SessionLocal() as session:
        doc = session.scalar(select(Document).where(Document.source_url == "https://example.com/doc1"))
        v1_id = doc.current_version_id

    # 第二次导入不同内容
    with SessionLocal() as session:
        result2 = import_package(session, package2)
        session.commit()
    assert result2.status == "imported"
    assert result2.version_status == "updated"

    # 验证：两个版本都存在，current_version_id 指向最新
    with SessionLocal() as session:
        doc = session.scalar(select(Document).where(Document.source_url == "https://example.com/doc1"))
        versions = session.scalars(select(DocumentVersion).where(DocumentVersion.document_id == doc.id)).all()

        assert len(versions) == 2
        assert doc.current_version_id != v1_id

        # 旧版本仍可查询
        old_version = session.get(DocumentVersion, v1_id)
        assert old_version is not None
        assert old_version.cleaned_content == content_v1

        # 新版本是当前版本
        new_version = session.get(DocumentVersion, doc.current_version_id)
        assert new_version is not None
        assert new_version.cleaned_content == content_v2

        # 旧版本的 chunk 也保留
        old_chunks = session.scalars(
            select(DocumentChunk).where(DocumentChunk.document_version_id == v1_id)
        ).all()
        assert len(old_chunks) > 0


def test_replace_existing_deletes_old_versions():
    """测试 replace_existing=True：删除旧版本"""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)

    content_v1 = "第一条 内容A"
    content_v2 = "第一条 内容B（修订）"

    package1 = _create_minimal_package("doc1", "https://example.com/doc1", content_v1, "v1")
    package2 = _create_minimal_package("doc1", "https://example.com/doc1", content_v2, "v2")

    # 第一次导入
    with SessionLocal() as session:
        import_package(session, package1)
        session.commit()

    # 第二次导入，使用 replace_existing=True
    with SessionLocal() as session:
        result = import_package(session, package2, replace_existing=True)
        session.commit()
    assert result.version_status == "updated"

    # 验证：旧版本已删除（只剩 1 个版本，且是新版本）
    with SessionLocal() as session:
        doc = session.scalar(select(Document).where(Document.source_url == "https://example.com/doc1"))
        versions = session.scalars(select(DocumentVersion).where(DocumentVersion.document_id == doc.id)).all()

        assert len(versions) == 1
        assert versions[0].version_key == "v2"
        assert versions[0].cleaned_content == content_v2

        # 旧版本的 version_key 不存在
        old_version = session.scalar(
            select(DocumentVersion).where(
                DocumentVersion.document_id == doc.id,
                DocumentVersion.version_key == "v1"
            )
        )
        assert old_version is None
