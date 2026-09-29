from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError

from app.db.import_service import import_package
from app.db.sql_models import Document, DocumentChunk, DocumentVersion, ImportRecord
from conftest import build_validated_package, create_test_session


def test_import_package_handles_new_unchanged_and_updated_versions() -> None:
    session = create_test_session()

    first = import_package(session, build_validated_package())
    repeated = import_package(session, build_validated_package(package_id="pkg-2"))
    updated = import_package(
        session,
        build_validated_package(
            package_id="pkg-3",
            version_key="ver-b",
            content_hash="hash-b",
        ),
    )

    assert first.status == "imported"
    assert first.version_status == "new"
    assert repeated.status == "already_imported"
    assert repeated.version_status == "unchanged"
    assert updated.status == "imported"
    assert updated.version_status == "updated"
    assert session.scalar(select(func.count()).select_from(Document)) == 1
    assert session.scalar(select(func.count()).select_from(DocumentVersion)) == 2
    assert session.scalar(select(func.count()).select_from(DocumentChunk)) == 4
    current_document = session.scalar(select(Document).where(Document.document_key == "doc-a"))
    assert current_document is not None
    assert current_document.current_version_id is not None


def test_repeated_package_id_is_idempotent() -> None:
    session = create_test_session()

    imported = import_package(session, build_validated_package(package_id="pkg-1"))
    repeated = import_package(session, build_validated_package(package_id="pkg-1"))

    assert imported.status == "imported"
    assert repeated.status == "already_imported"
    assert session.scalar(select(func.count()).select_from(ImportRecord)) == 1
    assert session.scalar(select(func.count()).select_from(DocumentVersion)) == 1


def test_failed_import_rolls_back_business_rows_and_records_failure() -> None:
    session = create_test_session()
    broken = build_validated_package()
    broken.data.chunks[0].__dict__["chunk_id"] = "duplicated"
    broken.data.chunks[1].__dict__["chunk_id"] = "duplicated"

    result = import_package(session, broken)

    assert result.status == "failed"
    assert "第一条 测试正文" not in (result.error_summary or "")
    assert session.scalar(select(func.count()).select_from(Document)) == 0
    assert session.scalar(select(func.count()).select_from(DocumentVersion)) == 0
    assert session.scalar(select(func.count()).select_from(DocumentChunk)) == 0
    failed_record = session.scalar(select(ImportRecord).where(ImportRecord.package_id == "pkg-1"))
    assert failed_record is not None
    assert failed_record.status == "failed"
    assert failed_record.error_summary is not None
