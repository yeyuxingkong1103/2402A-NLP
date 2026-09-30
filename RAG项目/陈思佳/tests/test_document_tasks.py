from pathlib import Path
from unittest.mock import Mock

from src.edu_rag_ingest.ingestion.document_service import DocumentService, DocumentStore


def test_upload_record_starts_in_pending_state(tmp_path: Path):
    service = DocumentService.__new__(DocumentService)
    service.store = DocumentStore(tmp_path / "documents.json")
    service.upload_dir = tmp_path

    record = service.save_upload("课程标准.md", "阅读教学要求".encode("utf-8"))

    assert record["status"] == "待解析"
    assert Path(record["file_path"]).exists()
    assert service.store.get(record["id"])["file_name"] == "课程标准.md"


def test_task_runner_submits_document_processing():
    from src.edu_rag_ingest.ingestion.document_tasks import DocumentTaskRunner

    runner = DocumentTaskRunner(max_workers=1)
    task = Mock(return_value={"status": "已入库"})

    future = runner.submit(task, "doc-1")

    assert future.result(timeout=2) == {"status": "已入库"}
    task.assert_called_once_with("doc-1")
    runner.executor.shutdown(wait=True)
