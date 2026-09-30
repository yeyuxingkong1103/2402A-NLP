from datetime import UTC, datetime
import json
from pathlib import Path
import shutil
import uuid

from app.pipeline.package_models import (
    CHUNKS_FILE,
    CRAWLS_FILE,
    DOCUMENTS_FILE,
    VERSIONS_FILE,
    PackageData,
    PackageFormatError,
)
from app.pipeline.hashing import file_sha256
from app.pipeline.package_validator import MANIFEST_FILE, SCHEMA_VERSION, validate_package


PIPELINE_VERSION = "0.1.0"
_RECORD_FILES = (DOCUMENTS_FILE, VERSIONS_FILE, CHUNKS_FILE, CRAWLS_FILE)


def publish_package(
    package_directory,
    data: PackageData,
    raw_content: bytes,
    package_id: str,
    created_at: str | None = None,
) -> Path:
    final_directory = Path(package_directory)
    if final_directory.exists():
        raise PackageFormatError("package directory already exists")

    temporary_directory = final_directory.parent / f".{final_directory.name}.{uuid.uuid4().hex}.tmp"
    try:
        data.write_records(temporary_directory)
        raw_file_path = _safe_raw_path(temporary_directory, data.version.raw_file_path)
        raw_file_path.parent.mkdir(parents=True, exist_ok=True)
        raw_file_path.write_bytes(raw_content)
        _write_manifest(temporary_directory, data, package_id, created_at)
        validate_package(temporary_directory)
        temporary_directory.replace(final_directory)
    except Exception as error:
        if temporary_directory.exists():
            shutil.rmtree(temporary_directory, ignore_errors=True)
        if isinstance(error, PackageFormatError):
            raise
        raise PackageFormatError("failed to publish package") from error
    return final_directory


def _write_manifest(
    package_directory: Path,
    data: PackageData,
    package_id: str,
    created_at: str | None,
) -> None:
    raw_path = data.version.raw_file_path
    file_names = (*_RECORD_FILES, raw_path)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "package_id": package_id,
        "created_at": created_at or datetime.now(UTC).isoformat(),
        "source_id": data.crawl.source_id,
        "source_url": data.crawl.source_url,
        "record_counts": {
            DOCUMENTS_FILE: 1,
            VERSIONS_FILE: 1,
            CHUNKS_FILE: len(data.chunks),
            CRAWLS_FILE: 1,
        },
        "file_hashes": {
            file_name: file_sha256(package_directory / file_name)
            for file_name in file_names
        },
    }
    (package_directory / MANIFEST_FILE).write_text(
        json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
    )


def _safe_raw_path(package_directory: Path, raw_file_path: str) -> Path:
    raw_path = Path(raw_file_path)
    if raw_path.is_absolute() or ".." in raw_path.parts:
        raise PackageFormatError("invalid field raw_file_path")
    return package_directory / raw_path

