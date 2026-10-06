import pytest
from pydantic import ValidationError

from backend.app.models import BuildStep, BuildTask, Chunk, Document, DocumentStatus


def test_chunk_requires_page_and_category():
    chunk = Chunk(
        chunk_id="chunk-1",
        document_id="doc-1",
        page=3,
        category="正文",
        text="这是真实 PDF 中的一段内容。",
        source_span="page=3:block=2",
    )

    assert chunk.page == 3
    assert chunk.category == "正文"


def test_chunk_rejects_invalid_page():
    with pytest.raises(ValidationError):
        Chunk(
            chunk_id="chunk-1",
            document_id="doc-1",
            page=0,
            category="正文",
            text="内容",
            source_span="page=0:block=1",
        )


def test_build_task_progress_matches_step():
    task = BuildTask.new(document_id="doc-1")
    running = task.mark_running(BuildStep.CHUNKING, 50)

    assert running.step == BuildStep.CHUNKING
    assert running.progress == 50
    assert running.status == "running"


def test_document_default_status_uploaded():
    document = Document.new(file_name="a.pdf", file_path="data/a.pdf")

    assert document.status == DocumentStatus.UPLOADED
    assert document.content_hash is None


def test_document_new_keeps_content_hash():
    document = Document.new(file_name="a.pdf", file_path="data/a.pdf", content_hash="abc123")

    assert document.content_hash == "abc123"


def test_document_content_hash_defaults_on_validate():
    raw = {
        "document_id": "doc-1",
        "file_name": "a.pdf",
        "file_path": "data/a.pdf",
        "status": "uploaded",
        "created_at": "2026-09-01T00:00:00Z",
    }

    document = Document.model_validate(raw)

    assert document.content_hash is None
