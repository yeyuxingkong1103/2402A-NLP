from backend.app.models import BuildTask, Chunk, Document
from backend.app.storage import JsonStateStore


def test_store_round_trips_document_and_task(tmp_path):
    store = JsonStateStore(tmp_path / "state.json")
    document = Document.new(file_name="a.pdf", file_path="data/a.pdf")
    task = BuildTask.new(document.document_id)

    store.save_document(document)
    store.save_task(task)

    assert store.get_document(document.document_id) == document
    assert store.get_task(task.task_id) == task


def test_store_lists_chunks_by_document(tmp_path):
    store = JsonStateStore(tmp_path / "state.json")
    chunk = Chunk(
        chunk_id="chunk-1",
        document_id="doc-1",
        page=1,
        category="标题",
        text="内容",
        source_span="page=1:block=1",
    )

    store.save_chunks([chunk])

    assert store.list_chunks(document_id="doc-1") == [chunk]
