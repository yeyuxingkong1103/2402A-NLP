"""pytest 共享夹具和测试工具函数。

禁止测试文件之间互相 import；所有跨文件复用的函数集中到这里。
"""

# 测试隔离：覆盖 Redis 库号为 9（必须在导入 app.* 之前设置）
import os
os.environ["REDIS_URL"] = "redis://127.0.0.1:6379/9"

import json
from hashlib import sha256
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.db.base import Base
from app.pipeline.package_models import (
    CHUNKS_FILE,
    CRAWLS_FILE,
    DOCUMENTS_FILE,
    VERSIONS_FILE,
    ChunkEntry,
    CrawlEntry,
    DocumentEntry,
    PackageData,
    VersionEntry,
)
from app.pipeline.package_validator import MANIFEST_FILE, SCHEMA_VERSION, ValidatedPackage, PackageManifest
from app.pipeline.package_writer import PIPELINE_VERSION


def create_test_session() -> Session:
    """创建 SQLite 内存测试会话。"""
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    return Session(engine)


def build_validated_package(
    *,
    package_id: str = "pkg-1",
    source_url: str = "https://example.com/law/a",
    document_key: str = "doc-a",
    version_key: str = "ver-a",
    content_hash: str = "hash-a",
) -> ValidatedPackage:
    """构造用于测试的已校验数据包。"""
    data = PackageData(
        document=DocumentEntry(
            document_id=document_key,
            source_url=source_url,
            title="测试法规",
        ),
        version=VersionEntry(
            version_id=version_key,
            document_id=document_key,
            content_hash=content_hash,
            media_type="text/html",
            cleaned_content="第一条 测试正文。",
            raw_file_path="raw/a.html",
        ),
        chunks=(
            ChunkEntry(
                chunk_id=f"{version_key}-parent",
                version_id=version_key,
                parent_chunk_id=None,
                chunk_type="parent",
                article_number="第一条",
                sequence=1,
                content="第一条 测试正文。",
                retrieval_text="测试法规 第一条 测试正文。",
            ),
            ChunkEntry(
                chunk_id=f"{version_key}-child",
                version_id=version_key,
                parent_chunk_id=f"{version_key}-parent",
                chunk_type="child",
                article_number="第一条",
                sequence=2,
                content="第一条 测试正文。",
                retrieval_text="测试法规 第一条 测试正文。",
            ),
        ),
        crawl=CrawlEntry(
            crawl_id=f"crawl-{content_hash[:24]}",
            source_id="source-test",
            source_url=source_url,
            http_status=200,
            content_hash=content_hash,
            raw_file_path="raw/a.html",
            collected_at="2026-09-16T00:00:00+00:00",
        ),
    )
    manifest = PackageManifest(
        schema_version=SCHEMA_VERSION,
        pipeline_version=PIPELINE_VERSION,
        package_id=package_id,
        created_at="2026-09-16T00:00:00+00:00",
        source_id="source-test",
        source_url=source_url,
        record_counts={},
        file_hashes={},
    )
    return ValidatedPackage(manifest=manifest, data=data)


def write_package(package_directory: Path) -> None:
    """写入完整的测试数据包到指定目录。"""
    raw_content = "<html><body>第一条 测试正文。</body></html>".encode("utf-8")
    content_hash = sha256(raw_content).hexdigest()
    data = PackageData(
        document=DocumentEntry(
            document_id="doc-cli",
            source_url="https://example.com/cli",
            title="CLI 测试法规",
        ),
        version=VersionEntry(
            version_id="ver-cli",
            document_id="doc-cli",
            content_hash=content_hash,
            media_type="text/html",
            cleaned_content="第一条 测试正文。",
            raw_file_path="raw/cli.html",
        ),
        chunks=(
            ChunkEntry(
                chunk_id="chunk-cli-parent",
                version_id="ver-cli",
                parent_chunk_id=None,
                chunk_type="parent",
                article_number="第一条",
                sequence=1,
                content="第一条 测试正文。",
                retrieval_text="CLI 测试法规 第一条 测试正文。",
            ),
        ),
        crawl=CrawlEntry(
            crawl_id="crawl-cli",
            source_id="source-cli",
            source_url="https://example.com/cli",
            http_status=200,
            content_hash=content_hash,
            raw_file_path="raw/cli.html",
            collected_at="2026-09-16T00:00:00+00:00",
        ),
    )
    data.write_records(package_directory)
    raw_path = package_directory / "raw" / "cli.html"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_bytes(raw_content)

    def file_hash(path: Path) -> str:
        return "sha256:" + sha256(path.read_bytes()).hexdigest()

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "package_id": "pkg-cli",
        "created_at": "2026-09-16T00:00:00+00:00",
        "source_id": "source-cli",
        "source_url": "https://example.com/cli",
        "record_counts": {
            DOCUMENTS_FILE: 1,
            VERSIONS_FILE: 1,
            CHUNKS_FILE: 1,
            CRAWLS_FILE: 1,
        },
        "file_hashes": {
            DOCUMENTS_FILE: file_hash(package_directory / DOCUMENTS_FILE),
            VERSIONS_FILE: file_hash(package_directory / VERSIONS_FILE),
            CHUNKS_FILE: file_hash(package_directory / CHUNKS_FILE),
            CRAWLS_FILE: file_hash(package_directory / CRAWLS_FILE),
        },
    }
    import json
    manifest_path = package_directory / MANIFEST_FILE
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


class FakeRedis:
    """轻量假 Redis：内存字典 + 集合，覆盖 SessionStore 用到的最小接口。

    集中到 conftest：认证落库后多个测试文件需要为 SessionStore 提供替身；
    禁止测试文件互相 import，故放共享层（test_session_store 内有独立完整版，不受影响）。
    """

    def __init__(self) -> None:
        self.data: dict[str, str] = {}
        self.sets: dict[str, set[str]] = {}

    def set(self, key: str, value: str, ex: int | None = None) -> None:
        self.data[key] = value

    def get(self, key: str) -> str | None:
        return self.data.get(key)

    def delete(self, *keys: str) -> int:
        count = 0
        for key in keys:
            if self.data.pop(key, None) is not None:
                count += 1
            if self.sets.pop(key, None) is not None:
                count += 1
        return count

    def sadd(self, key: str, *members: str) -> int:
        self.sets.setdefault(key, set()).update(members)
        return len(members)

    def srem(self, key: str, *members: str) -> int:
        bucket = self.sets.get(key, set())
        removed = 0
        for member in members:
            if member in bucket:
                bucket.discard(member)
                removed += 1
        return removed

    def smembers(self, key: str) -> "set[str]":
        return set(self.sets.get(key, set()))

    def expire(self, key: str, ttl: int) -> bool:
        return key in self.data or key in self.sets


def approve_pending_versions(session) -> None:
    """测试辅助：把待审核版本置为已发布（模拟审核通过）。

    阶段6 起新导入版本一律 pending_review，索引与检索只认 approved；
    需要"导入 → 索引/可检索"的测试先调用本函数模拟审核通过。
    """
    from sqlalchemy import update

    from app.db.sql_models import DocumentVersion

    session.execute(update(DocumentVersion).values(version_status="approved"))
    session.commit()
