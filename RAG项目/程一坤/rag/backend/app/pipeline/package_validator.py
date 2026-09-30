from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path

from app.pipeline.hashing import file_sha256

from app.pipeline.package_models import (
    CHUNKS_FILE,
    CRAWLS_FILE,
    DOCUMENTS_FILE,
    VERSIONS_FILE,
    PackageData,
    PackageFormatError,
)


MANIFEST_FILE = "manifest.json"
SCHEMA_VERSION = "1.0.0"
_RECORD_FILES = (DOCUMENTS_FILE, VERSIONS_FILE, CHUNKS_FILE, CRAWLS_FILE)


class PackageValidationError(PackageFormatError):
    pass


@dataclass(frozen=True)
class PackageManifest:
    schema_version: str
    pipeline_version: str
    package_id: str
    created_at: str
    source_id: str
    source_url: str
    record_counts: dict[str, int]
    file_hashes: dict[str, str]


@dataclass(frozen=True)
class ValidatedPackage:
    manifest: PackageManifest
    data: PackageData


def validate_package(package_directory: Path) -> ValidatedPackage:
    root = Path(package_directory)
    try:
        _require_files(root)
        manifest = _read_manifest(root)
        data = _read_records(root)
        _validate_manifest(root, manifest, data)
        _validate_references(root, data)
    except PackageValidationError:
        raise
    except PackageFormatError as error:
        raise PackageValidationError(str(error)) from error
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PackageValidationError("invalid package structure") from error
    return ValidatedPackage(manifest=manifest, data=data)


def _require_files(root: Path) -> None:
    for file_name in (MANIFEST_FILE, *_RECORD_FILES):
        if not (root / file_name).is_file():
            raise PackageValidationError(f"missing {file_name}")


def _read_manifest(root: Path) -> PackageManifest:
    try:
        value = json.loads((root / MANIFEST_FILE).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PackageValidationError(f"invalid JSON in {MANIFEST_FILE}") from error
    if not isinstance(value, dict):
        raise PackageValidationError(f"invalid record in {MANIFEST_FILE}")

    required_fields = {
        "schema_version", "pipeline_version", "package_id", "created_at",
        "source_id", "source_url", "record_counts", "file_hashes",
    }
    if set(value) != required_fields:
        raise PackageValidationError(f"invalid fields in {MANIFEST_FILE}")
    if not isinstance(value["record_counts"], dict):
        raise PackageValidationError(f"invalid field record_counts in {MANIFEST_FILE}")
    if not isinstance(value["file_hashes"], dict):
        raise PackageValidationError(f"invalid field file_hashes in {MANIFEST_FILE}")
    for field_name in required_fields - {"record_counts", "file_hashes"}:
        if not isinstance(value[field_name], str):
            raise PackageValidationError(f"invalid field {field_name} in {MANIFEST_FILE}")
    return PackageManifest(**value)


def _read_records(root: Path) -> PackageData:
    try:
        return PackageData.read_records(root)
    except PackageFormatError as error:
        raise PackageValidationError(str(error)) from error


def _validate_manifest(root: Path, manifest: PackageManifest, data: PackageData) -> None:
    if manifest.schema_version != SCHEMA_VERSION:
        raise PackageValidationError(f"invalid field schema_version in {MANIFEST_FILE}")
    expected_counts = {
        DOCUMENTS_FILE: 1,
        VERSIONS_FILE: 1,
        CHUNKS_FILE: len(data.chunks),
        CRAWLS_FILE: 1,
    }
    if manifest.record_counts != expected_counts:
        raise PackageValidationError(f"invalid field record_counts in {MANIFEST_FILE}")
    for file_name, expected_hash in manifest.file_hashes.items():
        file_path = _validate_managed_path(root, file_name, "file_hashes")
        if not file_path.is_file():
            raise PackageValidationError(f"missing {Path(file_name).name}")
        if file_sha256(file_path) != expected_hash:
            raise PackageValidationError(f"invalid sha256 for {Path(file_name).name}")
    for file_name in _RECORD_FILES:
        if file_name not in manifest.file_hashes:
            raise PackageValidationError(f"missing hash for {file_name}")


def _validate_references(root: Path, data: PackageData) -> None:
    if data.version.document_id != data.document.document_id:
        raise PackageValidationError(f"invalid field document_id in {VERSIONS_FILE}")
    chunk_ids: set[str] = set()
    for chunk in data.chunks:
        if chunk.version_id != data.version.version_id:
            raise PackageValidationError(f"invalid field version_id in {CHUNKS_FILE}")
        if chunk.chunk_id in chunk_ids:
            raise PackageValidationError(f"invalid field chunk_id in {CHUNKS_FILE}")
        chunk_ids.add(chunk.chunk_id)
    for chunk in data.chunks:
        if chunk.parent_chunk_id is not None and chunk.parent_chunk_id not in chunk_ids:
            raise PackageValidationError(f"invalid field parent_chunk_id in {CHUNKS_FILE}")

    version_raw_path = _validate_managed_path(root, data.version.raw_file_path, "raw_file_path")
    crawl_raw_path = _validate_managed_path(root, data.crawl.raw_file_path, "raw_file_path")
    if version_raw_path != crawl_raw_path:
        raise PackageValidationError(f"invalid field raw_file_path in {CRAWLS_FILE}")
    raw_hash = sha256(version_raw_path.read_bytes()).hexdigest()

    # 防篡改校验：CrawlEntry.content_hash 记录原始文件哈希，必须与本地原始文件重算结果一致
    # 不一致说明文件被篡改或传输损坏，必须拒绝
    if data.crawl.content_hash != raw_hash:
        raise PackageValidationError(f"invalid field content_hash in {CRAWLS_FILE}")

    # VersionEntry.content_hash 是语义哈希（清洗后正文的 SHA256），与原始文件哈希不同属正常
    # 只做格式校验，不与原始文件比对，因为它的用途是增量判定（检测语义是否变化）
    # 格式校验：必须是 64 字符的十六进制字符串
    if not isinstance(data.version.content_hash, str) or len(data.version.content_hash) != 64:
        raise PackageValidationError(f"invalid field content_hash in {VERSIONS_FILE}")
    try:
        int(data.version.content_hash, 16)
    except ValueError:
        raise PackageValidationError(f"invalid field content_hash in {VERSIONS_FILE}")

    # VersionEntry.chunk_fingerprint 是切块指纹（切块器版本 + 分块结构哈希）。
    # 老包无此字段（None，升级前产出），按"未知"处理放行；
    # 新包必须带，且格式与 content_hash 一致（64 位十六进制）。
    if data.version.chunk_fingerprint is not None:
        fingerprint = data.version.chunk_fingerprint
        if not isinstance(fingerprint, str) or len(fingerprint) != 64:
            raise PackageValidationError(f"invalid field chunk_fingerprint in {VERSIONS_FILE}")
        try:
            int(fingerprint, 16)
        except ValueError:
            raise PackageValidationError(f"invalid field chunk_fingerprint in {VERSIONS_FILE}")


def _validate_managed_path(root: Path, relative_path: str, field_name: str) -> Path:
    if not isinstance(relative_path, str) or not relative_path:
        raise PackageValidationError(f"invalid field {field_name}")
    path = Path(relative_path)
    if path.is_absolute() or ".." in path.parts:
        raise PackageValidationError(f"invalid field {field_name}")
    resolved_root = root.resolve()
    resolved_path = (root / path).resolve()
    if resolved_path != resolved_root and resolved_root not in resolved_path.parents:
        raise PackageValidationError(f"invalid field {field_name}")
    return resolved_path

