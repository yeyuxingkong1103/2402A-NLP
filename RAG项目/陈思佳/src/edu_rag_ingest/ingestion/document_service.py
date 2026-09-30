from __future__ import annotations

"""用户上传文档的生命周期服务，负责保存、解析、分块、向量化和删除。"""

import json
import re
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .chunker import TextChunker, write_chunks_jsonl
from .cleaner import clean_text
from ..config.config import AppConfig
from ..retrieval.embedding import LocalEmbeddingClient, build_embedding_text
from .mineru_client import MinerUClient
from ..retrieval.milvus_store import MilvusChunkStore
from .pipeline import build_metadata, setup_logging


_ALLOWED_SUFFIXES = {".pdf", ".docx", ".doc", ".md", ".txt"}
_SAFE_FILENAME = re.compile(r"[^\w.\-一-鿿]+")


class DocumentStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def list(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        try:
            records = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        if not isinstance(records, list):
            return []
        return sorted(records, key=lambda item: item.get("updated_at", ""), reverse=True)

    def get(self, document_id: str) -> dict[str, Any] | None:
        return next((item for item in self.list() if item.get("id") == document_id), None)

    def upsert(self, record: dict[str, Any]) -> dict[str, Any]:
        records = [item for item in self.list() if item.get("id") != record.get("id")]
        records.append(record)
        self.path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
        return record

    def delete(self, document_id: str) -> bool:
        records = self.list()
        remaining = [item for item in records if item.get("id") != document_id]
        if len(remaining) == len(records):
            return False
        self.path.write_text(json.dumps(remaining, ensure_ascii=False, indent=2), encoding="utf-8")
        return True


class DocumentService:
    def __init__(self, config: AppConfig, store: DocumentStore | None = None) -> None:
        self.config = config
        self.store = store or DocumentStore(Path("data/documents.json"))
        self.upload_dir = Path("data/raw/uploads")
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    def validate_filename(self, filename: str) -> str:
        safe_name = _SAFE_FILENAME.sub("_", Path(filename).name).strip("._")
        suffix = Path(safe_name).suffix.lower()
        if not safe_name or suffix not in _ALLOWED_SUFFIXES:
            allowed = ", ".join(sorted(_ALLOWED_SUFFIXES))
            raise ValueError(f"仅支持以下文件类型：{allowed}")
        return safe_name

    def save_upload(self, filename: str, content: bytes) -> dict[str, Any]:
        safe_name = self.validate_filename(filename)
        document_id = uuid.uuid4().hex
        source_path = self.upload_dir / f"{document_id}_{safe_name}"
        source_path.write_bytes(content)
        now = datetime.now(timezone.utc).isoformat()
        record = {
            "id": document_id,
            "title": Path(safe_name).stem,
            "file_name": safe_name,
            "file_path": str(source_path),
            "file_type": Path(safe_name).suffix.lower().lstrip("."),
            "status": "待解析",
            "chunk_count": 0,
            "created_at": now,
            "updated_at": now,
        }
        return self.store.upsert(record)

    def process(self, document_id: str, metadata_overrides: dict[str, str] | None = None) -> dict[str, Any]:
        record = self.store.get(document_id)
        if not record:
            raise FileNotFoundError("文档不存在")
        record = self._update_status(record, "解析中")
        try:
            source_path = Path(record["file_path"])
            mineru = MinerUClient(self.config.mineru)
            chunker = TextChunker(self.config.chunking)
            parsed_text = mineru.parse_or_read(source_path)
            parsed_path = mineru.save_parsed_text(source_path, parsed_text)
            cleaned = clean_text(parsed_text)
            metadata = build_metadata(
                self.config.metadata_defaults,
                str(source_path),
                parsed_path,
                str(record.get("title") or source_path.stem),
            )
            metadata.update(metadata_overrides or {})
            metadata["document_id"] = document_id
            chunks = chunker.split(document_id, cleaned, metadata)
            self._append_chunks(chunks)
            self._ingest_chunks(chunks)
            record.update(
                {
                    "status": "已入库",
                    "parsed_path": str(parsed_path),
                    "chunk_count": len(chunks),
                    "metadata": metadata,
                }
            )
            return self._update_status(record, "已入库")
        except Exception:
            self._update_status(record, "解析失败")
            raise

    def delete(self, document_id: str) -> bool:
        record = self.store.get(document_id)
        if not record:
            return False
        store = MilvusChunkStore(self.config.milvus)
        store.delete_document_chunks(document_id)
        self._remove_chunks(document_id)
        for key in ("file_path", "parsed_path"):
            path = Path(record[key]) if record.get(key) else None
            if path and path.exists():
                path.unlink()
        return self.store.delete(document_id)

    def _append_chunks(self, chunks: list[Any]) -> None:
        path = self.config.chunking.output_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as file:
            for chunk in chunks:
                file.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")

    def _remove_chunks(self, document_id: str) -> None:
        path = self.config.chunking.output_path
        if not path.exists():
            return
        kept: list[str] = []
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                chunk = json.loads(line)
                if chunk.get("document_id") != document_id:
                    kept.append(line)
        path.write_text("".join(kept), encoding="utf-8")

    def _ingest_chunks(self, chunks: list[Any]) -> None:
        if not chunks:
            return
        embedding_client = LocalEmbeddingClient(self.config.embedding)
        milvus = MilvusChunkStore(self.config.milvus)
        milvus.ensure_collection()
        rows = [asdict(chunk) for chunk in chunks]
        existing_ids = milvus.existing_chunk_ids([row["chunk_id"] for row in rows])
        pending = [row for row in rows if row["chunk_id"] not in existing_ids]
        if not pending:
            return
        vectors = embedding_client.encode(
            [build_embedding_text(row["content"], row.get("metadata", {})) for row in pending]
        )
        milvus.insert_chunks(pending, vectors)

    def _update_status(self, record: dict[str, Any], status: str) -> dict[str, Any]:
        record["status"] = status
        record["updated_at"] = datetime.now(timezone.utc).isoformat()
        return self.store.upsert(record)
