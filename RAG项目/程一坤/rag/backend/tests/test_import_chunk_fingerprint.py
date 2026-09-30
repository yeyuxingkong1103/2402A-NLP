"""导入幂等与切块指纹的契约测试。

对应 2.2 潜伏 bug 修复（两项根因）：
- 根因 A：unchanged 分支重复插入 import_records（package_id 有 UNIQUE 约束，
  同包 replace_existing 重导必然撞 1062）→ _add_import_record 必须幂等
- 根因 B：增量判定只看 content_hash（正文语义哈希），切块变化不触发重建
  → VersionEntry/document_versions 增加 chunk_fingerprint，双比较判定：
  · 两者都相同 → unchanged（跳过）
  · content_hash 同、指纹不同 → updated，原因记 chunking_changed，
    重建分块，旧版本保留为历史
  · content_hash 不同 → updated（保持现有行为）
- 存量兼容：老版本行 chunk_fingerprint 为空按"未知"处理，
  不得因为为空就一律判 updated（否则升级后第一次重跑会全量重建）
"""
import hashlib
from sqlalchemy import create_engine, select, func
from sqlalchemy.orm import sessionmaker
from app.db.sql_models import (
    Base,
    Document,
    DocumentVersion,
    DocumentChunk,
    ImportRecord,
)
from app.db.import_service import import_package
from app.pipeline.package_validator import ValidatedPackage, PackageManifest
from app.pipeline.package_models import (
    PackageData,
    DocumentEntry,
    VersionEntry,
    ChunkEntry,
    CrawlEntry,
)


def _create_package(
    document_id: str,
    source_url: str,
    content: str,
    version_id: str,
    package_id: str,
    chunk_fingerprint: str | None,
    title: str = "测试文档",
) -> ValidatedPackage:
    """构造测试用 ValidatedPackage；指纹与 version_id 均可显式指定。"""
    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

    data = PackageData(
        document=DocumentEntry(
            document_id=document_id,
            source_url=source_url,
            title=title,
        ),
        version=VersionEntry(
            version_id=version_id,
            document_id=document_id,
            content_hash=content_hash,
            media_type="text/html",
            cleaned_content=content,
            raw_file_path="test.html",
            # 切块指纹：pipeline 打包时写入；None 模拟老版本包（无此信息）
            chunk_fingerprint=chunk_fingerprint,
        ),
        chunks=(
            ChunkEntry(
                chunk_id=f"{version_id}_chunk1",
                version_id=version_id,
                parent_chunk_id=None,
                chunk_type="article",
                article_number="第一条",
                sequence=0,
                content=content,
                retrieval_text=content,
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

    manifest = PackageManifest(
        schema_version="1.0.0",
        pipeline_version="test",
        package_id=package_id,
        created_at="2024-01-01T00:00:00Z",
        source_id=document_id,
        source_url=source_url,
        record_counts={"documents": 1, "versions": 1, "chunks": 1, "crawls": 1},
        file_hashes={},
    )
    return ValidatedPackage(manifest=manifest, data=data)


def _make_session_factory() -> sessionmaker:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)


CONTENT = "第一条 为了完善劳动合同制度，保护劳动者合法权益，特制定本法。"
FINGERPRINT_V1 = "a" * 64
FINGERPRINT_V2 = "b" * 64


def test_same_package_reimport_with_replace_existing_is_idempotent():
    """根因 A：同一数据包 replace_existing 重导不得撞 import_records 唯一约束。

    之前的行为：replace_existing 绕过入口幂等检查 → unchanged 分支再插一行
    import_records → package_id UNIQUE 冲突 → 整个包判 failed（1062）。
    """
    SessionLocal = _make_session_factory()
    package = _create_package(
        "doc1",
        "https://example.com/labor-law",
        CONTENT,
        version_id="v1",
        package_id="pkg_labor_law_v1",
        chunk_fingerprint=FINGERPRINT_V1,
    )

    with SessionLocal() as session:
        result1 = import_package(session, package, replace_existing=False)
        session.commit()
    assert result1.status == "imported"

    # 同一个包再来一次（replace_existing=True 走完整链路，是最严苛场景）
    with SessionLocal() as session:
        result2 = import_package(session, package, replace_existing=True)
        session.commit()

    # 第二次不得 failed / 不得报唯一键冲突
    assert result2.status == "already_imported", result2
    assert result2.version_status == "unchanged"

    # 行数全部不变：版本 1、导入记录 1、分块 1
    with SessionLocal() as session:
        assert session.scalar(select(func.count(DocumentVersion.id))) == 1
        assert session.scalar(select(func.count(ImportRecord.id))) == 1
        assert session.scalar(select(func.count(DocumentChunk.id))) == 1
        record = session.scalar(select(ImportRecord))
        assert record.status == "already_imported"


def test_chunking_change_creates_new_version_and_keeps_history():
    """根因 B：正文相同、切块指纹不同 → updated（chunking_changed），旧版本保留。"""
    SessionLocal = _make_session_factory()

    package_v1 = _create_package(
        "doc1",
        "https://example.com/labor-law",
        CONTENT,
        version_id="ver-aaa",
        package_id="pkg_labor_law_v1",
        chunk_fingerprint=FINGERPRINT_V1,
    )
    # 正文不变，仅切块方式变化（不同指纹 + 不同 version_id）
    package_v2 = _create_package(
        "doc1",
        "https://example.com/labor-law",
        CONTENT,
        version_id="ver-bbb",
        package_id="pkg_labor_law_v2",
        chunk_fingerprint=FINGERPRINT_V2,
    )

    with SessionLocal() as session:
        result1 = import_package(session, package_v1)
        session.commit()
    assert result1.status == "imported"
    with SessionLocal() as session:
        doc = session.scalar(select(Document).where(Document.source_url == "https://example.com/labor-law"))
        old_version_id = doc.current_version_id

    with SessionLocal() as session:
        result2 = import_package(session, package_v2, replace_existing=False)
        session.commit()

    # 必须判 updated 且原因记为 chunking_changed，而不是 unchanged
    assert result2.status == "imported", result2
    assert result2.version_status == "updated"
    assert result2.reason == "chunking_changed"

    with SessionLocal() as session:
        doc = session.scalar(select(Document).where(Document.source_url == "https://example.com/labor-law"))
        # 两个版本并存：旧版本保留为历史，current 指向新版本
        versions = session.scalars(
            select(DocumentVersion).where(DocumentVersion.document_id == doc.id)
        ).all()
        assert len(versions) == 2
        assert doc.current_version_id != old_version_id

        old_version = session.get(DocumentVersion, old_version_id)
        assert old_version is not None
        assert old_version.chunk_fingerprint == FINGERPRINT_V1
        # 旧版本的分块仍可查（历史可回溯）
        old_chunks = session.scalars(
            select(DocumentChunk).where(DocumentChunk.document_version_id == old_version_id)
        ).all()
        assert len(old_chunks) == 1

        new_version = session.get(DocumentVersion, doc.current_version_id)
        assert new_version.chunk_fingerprint == FINGERPRINT_V2
        assert session.scalar(select(func.count(ImportRecord.id))) == 2


def test_null_fingerprint_treated_as_unknown_keeps_unchanged():
    """存量兼容：老版本行指纹为空按"未知"处理，不得触发全量重建。"""
    SessionLocal = _make_session_factory()

    # 模拟存量数据：老版本没有指纹（chunk_fingerprint=None）
    legacy_package = _create_package(
        "doc1",
        "https://example.com/labor-law",
        CONTENT,
        version_id="ver-legacy",
        package_id="pkg_labor_law_legacy",
        chunk_fingerprint=None,
    )
    with SessionLocal() as session:
        result1 = import_package(session, legacy_package)
        session.commit()
    assert result1.status == "imported"

    # 新管道重新打包：内容相同但带上了指纹
    new_package = _create_package(
        "doc1",
        "https://example.com/labor-law",
        CONTENT,
        version_id="ver-new",
        package_id="pkg_labor_law_new",
        chunk_fingerprint=FINGERPRINT_V1,
    )
    with SessionLocal() as session:
        result2 = import_package(session, new_package, replace_existing=False)
        session.commit()

    # 指纹为空的旧记录 → "未知" → unchanged，不重建
    assert result2.status == "already_imported", result2
    assert result2.version_status == "unchanged"
    with SessionLocal() as session:
        assert session.scalar(select(func.count(DocumentVersion.id))) == 1
