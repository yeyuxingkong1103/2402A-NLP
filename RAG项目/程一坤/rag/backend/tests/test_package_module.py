"""数据包模块测试：记录模型 / 完整性校验 / 原子发布。

数据包是离线采集与 MySQL 入库之间唯一的交接物，坏数据包一旦进库，
脏数据会顺着检索链路传到用户回答里，因此只覆盖真会出问题的行为。"""

import json
from hashlib import sha256
from pathlib import Path

import pytest

from app.pipeline.package_models import (
    CHUNKS_FILE,
    CRAWLS_FILE,
    DOCUMENTS_FILE,
    VERSIONS_FILE,
    ChunkEntry,
    CrawlEntry,
    DocumentEntry,
    PackageData,
    PackageFormatError,
    VersionEntry,
)
from app.pipeline.package_validator import (
    MANIFEST_FILE,
    SCHEMA_VERSION,
    PackageValidationError,
    validate_package,
)
from app.pipeline.package_writer import PIPELINE_VERSION, publish_package

RAW_RELATIVE_PATH = "raw/doc-001.html"
RAW_CONTENT = "<html><body><p>第一条 测试正文。</p></body></html>".encode("utf-8")
JSONL_FILE_NAMES = (DOCUMENTS_FILE, VERSIONS_FILE, CHUNKS_FILE, CRAWLS_FILE)


def calculate_file_sha256(file_path: Path) -> str:
    """计算受管文件的 SHA-256；测试与实现共用同一口径。"""
    return "sha256:" + sha256(file_path.read_bytes()).hexdigest()


def build_chunk(
    chunk_id: str, parent_chunk_id: str | None, sequence: int, content: str
) -> ChunkEntry:
    """构造一个分块记录；检索文本按「标题 + 条号 + 正文」拼装。"""
    return ChunkEntry(
        chunk_id=chunk_id,
        version_id="ver-001",
        parent_chunk_id=parent_chunk_id,
        chunk_type="parent" if parent_chunk_id is None else "child",
        article_number="第一条",
        sequence=sequence,
        content=content,
        retrieval_text=f"临时法规 第一条 {content}",
    )


def build_valid_package_data() -> PackageData:
    """构造字段互相自洽的最小合法数据包，作为各用例的基线。"""
    content_hash = sha256(RAW_CONTENT).hexdigest()
    return PackageData(
        document=DocumentEntry(
            document_id="doc-001", source_url="https://example.com/law/1", title="临时法规"),
        version=VersionEntry(
            version_id="ver-001", document_id="doc-001", content_hash=content_hash,
            media_type="text/html", cleaned_content="第一条 测试正文。",
            raw_file_path=RAW_RELATIVE_PATH),
        chunks=(
            build_chunk("doc-001-a1", None, 1, "第一条 测试正文。"),
            build_chunk("doc-001-a1-c1", "doc-001-a1", 2, "测试正文。"),
        ),
        crawl=CrawlEntry(
            crawl_id="crawl-001", source_id="temporary-source",
            source_url="https://example.com/law/1", http_status=200,
            content_hash=content_hash, raw_file_path=RAW_RELATIVE_PATH,
            collected_at="2026-09-16T00:00:00+00:00"),
    )


def replace_version_field(data: PackageData, **changes) -> PackageData:
    """返回只改动版本字段的数据包副本。"""
    return PackageData(
        document=data.document,
        version=VersionEntry(**{**data.version.__dict__, **changes}),
        chunks=data.chunks,
        crawl=data.crawl,
    )


def build_manifest_dictionary(package_directory: Path, data: PackageData) -> dict:
    """按清单契约组装 manifest 内容，供写入与重算共用。"""
    return {
        "schema_version": SCHEMA_VERSION,
        "pipeline_version": "0.1.0",
        "package_id": "pkg-temporary-001",
        "created_at": "2026-09-16T00:00:00+00:00",
        "source_id": data.crawl.source_id,
        "source_url": data.crawl.source_url,
        "record_counts": {
            DOCUMENTS_FILE: 1, VERSIONS_FILE: 1,
            CHUNKS_FILE: len(data.chunks), CRAWLS_FILE: 1,
        },
        "file_hashes": {
            name: calculate_file_sha256(package_directory / name)
            for name in (*JSONL_FILE_NAMES, RAW_RELATIVE_PATH)
        },
    }


def write_valid_package_directory(package_directory: Path) -> PackageData:
    """写一个完全合法的数据包目录，便于逐项篡改。"""
    data = build_valid_package_data()
    data.write_records(package_directory)
    raw_file_path = package_directory / RAW_RELATIVE_PATH
    raw_file_path.parent.mkdir(parents=True, exist_ok=True)
    raw_file_path.write_bytes(RAW_CONTENT)
    write_manifest_file(package_directory, data)
    return data


def write_manifest_file(package_directory: Path, data: PackageData) -> None:
    """写入清单；重算哈希后的场景也走这里。"""
    (package_directory / MANIFEST_FILE).write_text(
        json.dumps(build_manifest_dictionary(package_directory, data), ensure_ascii=False),
        encoding="utf-8",
    )


def rewrite_records_and_manifest(package_directory: Path, broken: PackageData) -> None:
    """把改动后的记录写回目录并重算清单。

    重算哈希是为了让校验器不能只靠哈希发现问题，
    必须靠引用、路径和内容一致性拦下来。
    """
    broken.write_records(package_directory)
    write_manifest_file(package_directory, broken)


def list_temporary_directories(parent_directory: Path) -> list[str]:
    """列出同级目录里疑似临时目录的条目（约定以 . 开头）。"""
    return sorted(
        entry.name for entry in parent_directory.iterdir()
        if entry.is_dir() and entry.name.startswith("."))


def test_valid_package_passes_validation(tmp_path):
    """结构完整、哈希匹配的数据包应通过校验。"""
    expected = write_valid_package_directory(tmp_path)

    validated = validate_package(tmp_path)

    assert validated.manifest.schema_version == SCHEMA_VERSION
    assert validated.manifest.package_id == "pkg-temporary-001"
    assert validated.data.document == expected.document
    assert validated.data.chunks == expected.chunks


@pytest.mark.parametrize("missing_file_name", [MANIFEST_FILE, CRAWLS_FILE])
def test_missing_file_is_rejected(tmp_path, missing_file_name):
    """缺少清单或任一必需记录文件时必须拒绝。"""
    write_valid_package_directory(tmp_path)
    (tmp_path / missing_file_name).unlink()

    with pytest.raises(PackageValidationError):
        validate_package(tmp_path)


def test_tampered_file_is_rejected(tmp_path):
    """JSONL 或原始文件被改动后哈希不一致，必须拒绝。"""
    write_valid_package_directory(tmp_path)
    with (tmp_path / CHUNKS_FILE).open("a", encoding="utf-8") as output:
        output.write(json.dumps({"chunk_id": "injected"}, ensure_ascii=False) + "\n")

    with pytest.raises(PackageValidationError):
        validate_package(tmp_path)

    write_valid_package_directory(tmp_path)
    (tmp_path / RAW_RELATIVE_PATH).write_bytes(b"tampered")

    with pytest.raises(PackageValidationError):
        validate_package(tmp_path)


def test_unsafe_raw_path_is_rejected(tmp_path):
    """raw 路径逃逸数据包时必须拒绝。"""
    data = write_valid_package_directory(tmp_path)
    rewrite_records_and_manifest(
        tmp_path, replace_version_field(data, raw_file_path="../secret.html")
    )
    (tmp_path.parent / "secret.html").write_bytes(RAW_CONTENT)

    with pytest.raises(PackageValidationError):
        validate_package(tmp_path)


@pytest.mark.parametrize("broken_chunk_kind", ["duplicated", "orphaned"])
def test_broken_chunk_reference_is_rejected(tmp_path, broken_chunk_kind):
    """chunk_id 重复，或子块引用的父块不存在时，都必须拒绝。"""
    data = write_valid_package_directory(tmp_path)
    extra_chunk = (
        data.chunks[1]
        if broken_chunk_kind == "duplicated"
        else build_chunk("doc-001-a9-c1", "doc-001-a9", 3, "孤儿子块。")
    )
    broken_data = PackageData(
        document=data.document, version=data.version,
        chunks=data.chunks + (extra_chunk,), crawl=data.crawl)
    rewrite_records_and_manifest(tmp_path, broken_data)

    with pytest.raises(PackageValidationError):
        validate_package(tmp_path)


@pytest.mark.parametrize(
    "changes", [{"document_id": "doc-404"}]
)
def test_version_inconsistent_with_package_is_rejected(tmp_path, changes):
    """版本归属的文档不存在时必须拒绝。

    注：VersionEntry.content_hash 是语义哈希（清洗后正文），用于增量判定，
    只做格式校验，不与原始文件比对。防篡改由 CrawlEntry.content_hash 负责。
    """
    data = write_valid_package_directory(tmp_path)
    rewrite_records_and_manifest(tmp_path, replace_version_field(data, **changes))

    with pytest.raises(PackageValidationError):
        validate_package(tmp_path)


def test_error_message_does_not_leak_content(tmp_path):
    """校验失败信息不能带正文，否则会顺着日志泄漏法律数据。"""
    leaked_text = "不得出现在错误信息中的正文"
    write_valid_package_directory(tmp_path)
    broken_chunk = dict(
        chunk_id="doc-001-a1", version_id="ver-001", parent_chunk_id=None,
        chunk_type="parent", article_number="第一条", sequence="not-an-integer",
        content=leaked_text, retrieval_text=leaked_text,
    )
    (tmp_path / CHUNKS_FILE).write_text(
        json.dumps(broken_chunk, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    with pytest.raises(PackageValidationError) as captured_error:
        validate_package(tmp_path)

    assert leaked_text not in str(captured_error.value)


def test_publish_creates_package_that_passes_validation(tmp_path):
    """发布成功后应生成完整数据包、能通过校验，且不留临时目录。"""
    package_directory = tmp_path / "pkg-001"

    published_directory = publish_package(
        package_directory, build_valid_package_data(), RAW_CONTENT,
        package_id="pkg-001", created_at="2026-09-16T01:02:03+00:00",
    )

    assert published_directory == package_directory
    assert (package_directory / MANIFEST_FILE).is_file()
    assert (package_directory / RAW_RELATIVE_PATH).read_bytes() == RAW_CONTENT
    for file_name in JSONL_FILE_NAMES:
        assert (package_directory / file_name).is_file()

    validated = validate_package(package_directory)
    assert validated.manifest.package_id == "pkg-001"
    assert validated.manifest.pipeline_version == PIPELINE_VERSION
    assert list_temporary_directories(tmp_path) == []


def test_publish_rejects_existing_directory_and_preserves_it(tmp_path):
    """目标目录已存在时拒绝发布，且不破坏已发布内容。"""
    package_directory = tmp_path / "pkg-002"
    package_directory.mkdir()
    existing_file = package_directory / "keep.txt"
    existing_file.write_text("已发布的数据包不可被覆盖", encoding="utf-8")

    with pytest.raises(PackageFormatError):
        publish_package(
            package_directory, build_valid_package_data(), RAW_CONTENT, package_id="pkg-002"
        )

    assert existing_file.read_text(encoding="utf-8") == "已发布的数据包不可被覆盖"
    assert not (package_directory / MANIFEST_FILE).exists()
    assert list_temporary_directories(tmp_path) == []


def test_publish_failure_leaves_no_final_or_temporary_directory(tmp_path):
    """发布中途失败时不能留下最终目录，也不能残留临时目录。"""
    # 版本归属的文档 ID 不存在，发布前的校验必须拦下
    broken_data = replace_version_field(build_valid_package_data(), document_id="doc-404")
    package_directory = tmp_path / "pkg-003"

    with pytest.raises(PackageFormatError):
        publish_package(package_directory, broken_data, RAW_CONTENT, package_id="pkg-003")

    assert not package_directory.exists()
    assert list_temporary_directories(tmp_path) == []
