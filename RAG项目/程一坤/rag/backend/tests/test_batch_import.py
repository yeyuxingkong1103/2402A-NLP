from pathlib import Path
from hashlib import sha256

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.db.batch_import import find_package_directories, import_packages
from app.db.base import Base
from app.db.sql_models import Document, DocumentChunk, DocumentVersion
from app.pipeline.package_writer import publish_package
from conftest import build_validated_package


def test_batch_import_discovers_standard_packages_and_is_idempotent(tmp_path: Path) -> None:
    """批量导入只处理标准包，并且重复执行不会产生重复业务数据。"""
    packages_root = tmp_path / "packages"
    packages_root.mkdir()
    raw_a = b"raw-a"
    package_a = build_validated_package(
        package_id="pkg-a",
        document_key="doc-a",
        version_key="ver-a",
        content_hash=sha256(raw_a).hexdigest(),
    )
    publish_package(
        packages_root / "pkg-a",
        package_a.data,
        raw_a,
        package_id="pkg-a",
    )
    raw_b = b"raw-b"
    package_b = build_validated_package(
        package_id="pkg-b",
        document_key="doc-b",
        version_key="ver-b",
        source_url="https://example.com/b",
        content_hash=sha256(raw_b).hexdigest(),
    )
    publish_package(
        packages_root / "pkg-b",
        package_b.data,
        raw_b,
        package_id="pkg-b",
    )
    # 旧 JSONL 目录没有 manifest，不应进入严格入库流程。
    (packages_root / "legacy").mkdir()
    (packages_root / "legacy" / "documents.jsonl").write_text("legacy", encoding="utf-8")

    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        first = import_packages(session, packages_root)
        second = import_packages(session, packages_root)

        assert [path.name for path in find_package_directories(packages_root)] == ["pkg-a", "pkg-b"]
        assert (first.total_packages, first.imported, first.failed) == (2, 2, 0)
        assert (second.total_packages, second.already_imported, second.failed) == (2, 2, 0)
        assert session.query(Document).count() == 2
        assert session.query(DocumentVersion).count() == 2
        assert session.query(DocumentChunk).count() == 4
