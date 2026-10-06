from pathlib import Path

import pytest

from src.edu_rag_ingest.ingestion.document_service import DocumentService, DocumentStore


def test_validate_filename_allows_supported_suffix(tmp_path: Path):
    service = DocumentService.__new__(DocumentService)
    service.store = DocumentStore(tmp_path / "documents.json")
    service.upload_dir = tmp_path

    assert service.validate_filename("课程标准.pdf") == "课程标准.pdf"


def test_validate_filename_rejects_unsupported_suffix(tmp_path: Path):
    service = DocumentService.__new__(DocumentService)
    service.store = DocumentStore(tmp_path / "documents.json")
    service.upload_dir = tmp_path

    with pytest.raises(ValueError, match="仅支持"):
        service.validate_filename("malware.exe")


def test_document_store_upsert_list_and_delete(tmp_path: Path):
    store = DocumentStore(tmp_path / "documents.json")
    record = {"id": "doc-1", "status": "待解析", "updated_at": "2026-01-01"}

    store.upsert(record)

    assert store.get("doc-1") == record
    assert store.list() == [record]
    assert store.delete("doc-1") is True
    assert store.get("doc-1") is None
    assert store.delete("doc-1") is False
