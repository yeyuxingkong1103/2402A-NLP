from dataclasses import asdict, dataclass, fields
from hashlib import sha256
import json
from pathlib import Path
from types import UnionType
from typing import Any, Union, get_args, get_origin, get_type_hints


DOCUMENTS_FILE = "documents.jsonl"
VERSIONS_FILE = "document_versions.jsonl"
CHUNKS_FILE = "document_chunks.jsonl"
CRAWLS_FILE = "crawl_records.jsonl"


class PackageFormatError(ValueError):
    pass


@dataclass(frozen=True)
class DocumentEntry:
    document_id: str
    source_url: str
    title: str


@dataclass(frozen=True)
class VersionEntry:
    version_id: str
    document_id: str
    content_hash: str
    media_type: str
    cleaned_content: str
    raw_file_path: str
    # 切块指纹：切块器版本 + 分块结果（条/款/项结构）的哈希，由打包阶段写入。
    # 只用于增量判定的"切块是否变化"维度，不参与 content_hash 的防篡改校验；
    # None 表示老版本数据包（升级前产出），导入侧按"未知"处理。
    chunk_fingerprint: str | None = None


@dataclass(frozen=True)
class ChunkEntry:
    chunk_id: str
    version_id: str
    parent_chunk_id: str | None
    chunk_type: str
    article_number: str | None
    sequence: int
    content: str
    retrieval_text: str
    # 款号与项号：绝大多数条文没有分款分项，因此给默认值 None，
    # 调用方不必为每个分块显式写 paragraph_number=None, item_number=None
    paragraph_number: str | None = None
    item_number: str | None = None


@dataclass(frozen=True)
class CrawlEntry:
    crawl_id: str
    source_id: str
    source_url: str
    http_status: int
    content_hash: str
    raw_file_path: str
    collected_at: str


@dataclass(frozen=True)
class PackageData:
    document: DocumentEntry
    version: VersionEntry
    chunks: tuple[ChunkEntry, ...]
    crawl: CrawlEntry

    def write_records(self, root: Path) -> None:
        if not self.chunks:
            raise PackageFormatError("package must contain at least one chunk")
        root.mkdir(parents=True, exist_ok=True)
        _write_jsonl(root / DOCUMENTS_FILE, [self.document])
        _write_jsonl(root / VERSIONS_FILE, [self.version])
        _write_jsonl(root / CHUNKS_FILE, self.chunks)
        _write_jsonl(root / CRAWLS_FILE, [self.crawl])

    @classmethod
    def read_records(cls, root: Path) -> "PackageData":
        document = _read_single(root / DOCUMENTS_FILE, DocumentEntry)
        version = _read_single(root / VERSIONS_FILE, VersionEntry)
        chunks = _read_many(root / CHUNKS_FILE, ChunkEntry)
        if not chunks:
            raise PackageFormatError(f"{CHUNKS_FILE} must contain at least one record")
        crawl = _read_single(root / CRAWLS_FILE, CrawlEntry)
        return cls(document=document, version=version, chunks=chunks, crawl=crawl)


def stable_document_id(source_url: str) -> str:
    return "doc-" + sha256(source_url.strip().encode("utf-8")).hexdigest()[:24]


def stable_version_id(document_id: str, content_hash: str) -> str:
    value = f"{document_id}:{content_hash}"
    return "ver-" + sha256(value.encode("utf-8")).hexdigest()[:24]


def _write_jsonl(path: Path, records: Any) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as output:
        for record in records:
            output.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def _read_single(path: Path, entry_type: type[Any]) -> Any:
    records = _read_many(path, entry_type)
    if len(records) != 1:
        raise PackageFormatError(f"{path.name} must contain exactly one record")
    return records[0]


def _read_many(path: Path, entry_type: type[Any]) -> tuple[Any, ...]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise PackageFormatError(f"unable to read {path.name}") from error

    records = []
    for line_number, line in enumerate(lines, start=1):
        try:
            value = json.loads(line)
        except (json.JSONDecodeError, TypeError) as error:
            raise PackageFormatError(
                f"invalid JSON in {path.name} at line {line_number}"
            ) from error
        records.append(_parse_entry(value, entry_type, path.name, line_number))
    return tuple(records)


def _parse_entry(
    value: Any,
    entry_type: type[Any],
    filename: str,
    line_number: int,
) -> Any:
    if not isinstance(value, dict):
        raise PackageFormatError(f"invalid record in {filename} at line {line_number}")

    expected_fields = {field.name for field in fields(entry_type)}
    if set(value) != expected_fields:
        raise PackageFormatError(f"invalid fields in {filename} at line {line_number}")

    type_hints = get_type_hints(entry_type)
    for field_name, expected_type in type_hints.items():
        if not _matches_type(value[field_name], expected_type):
            raise PackageFormatError(
                f"invalid type for {field_name} in {filename} at line {line_number}"
            )
    return entry_type(**value)


def _matches_type(value: Any, expected_type: Any) -> bool:
    origin = get_origin(expected_type)
    if origin in (Union, UnionType):
        return any(_matches_type(value, option) for option in get_args(expected_type))
    if expected_type is int:
        return type(value) is int
    return isinstance(value, expected_type)
