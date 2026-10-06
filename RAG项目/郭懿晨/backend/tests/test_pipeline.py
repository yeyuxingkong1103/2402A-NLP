from pathlib import Path

from backend.app.embeddings import FakeBgeM3Embedder
from backend.app.mineru import MinerUBlock
from backend.app.models import BuildStep, BuildTask, Document
from backend.app.pipeline import BuildPipeline
from backend.app.storage import JsonStateStore
from backend.app.vector_store import InMemoryVectorStore


class FakeParser:
    def parse_pdf(self, pdf_path: Path) -> list[MinerUBlock]:
        return [MinerUBlock(page=1, label="正文", text="真实内容", source_span="page=1:block=1")]


class FakePostProcessor:
    def process(self, pdf_path: Path, blocks: list[MinerUBlock]) -> list[MinerUBlock]:
        return [
            blocks[0].model_copy(
                update={
                    "raw_text": blocks[0].text,
                    "text": "| 列1 | 列2 |\n| --- | --- |\n| A | B |",
                    "vision_text": "| 列1 | 列2 |\n| --- | --- |\n| A | B |",
                    "vision_model": "fake-vlm",
                    "vision_applied": True,
                }
            )
        ]


def test_build_pipeline_updates_task_and_writes_chunks(tmp_path):
    store = JsonStateStore(tmp_path / "state.json")
    vector_store = InMemoryVectorStore()
    document = Document.new(file_name="a.pdf", file_path=str(tmp_path / "a.pdf"))
    task = BuildTask.new(document.document_id)
    store.save_document(document)
    store.save_task(task)

    pipeline = BuildPipeline(store, FakeParser(), FakeBgeM3Embedder(), vector_store)
    pipeline.run(task.task_id)

    saved_task = store.get_task(task.task_id)
    assert saved_task.step == BuildStep.COMPLETED
    assert saved_task.progress == 100
    assert store.list_chunks(document.document_id)[0].category == "正文"
    assert vector_store.list_records()[0].category == "正文"


def test_build_pipeline_uses_post_processor_before_chunking(tmp_path):
    store = JsonStateStore(tmp_path / "state.json")
    vector_store = InMemoryVectorStore()
    document = Document.new(file_name="a.pdf", file_path=str(tmp_path / "a.pdf"))
    task = BuildTask.new(document.document_id)
    store.save_document(document)
    store.save_task(task)

    pipeline = BuildPipeline(store, FakeParser(), FakeBgeM3Embedder(), vector_store, post_processor=FakePostProcessor())
    pipeline.run(task.task_id)

    chunk = store.list_chunks(document.document_id)[0]

    assert chunk.text == "| 列1 | 列2 |\n| --- | --- |\n| A | B |"
    assert chunk.category == "正文"
