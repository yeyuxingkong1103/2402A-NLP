from pathlib import Path
from typing import Protocol

from backend.app.chunking import build_chunks, clean_blocks
from backend.app.embeddings import TextEmbedder
from backend.app.mineru import MinerUBlock
from backend.app.models import BuildStep, DocumentStatus
from backend.app.storage import JsonStateStore
from backend.app.vector_store import VectorStore
from backend.app.mineru_vlm import MinerUVLMPostProcessor


class PdfParser(Protocol):
    """PDF 解析器协议。"""

    def parse_pdf(self, pdf_path: Path) -> list[MinerUBlock]:
        """解析 PDF。"""


class PostProcessor(Protocol):
    """块后处理器协议。"""

    def process(self, pdf_path: Path, blocks: list[MinerUBlock]) -> list[MinerUBlock]:
        """处理解析后的块。"""


class BuildPipeline:
    """构建任务流水线。"""

    def __init__(
        self,
        store: JsonStateStore,
        parser: PdfParser,
        embedder: TextEmbedder,
        vector_store: VectorStore,
        post_processor: PostProcessor | None = None,
    ) -> None:
        self.store = store
        self.parser = parser
        self.embedder = embedder
        self.vector_store = vector_store
        self.post_processor = post_processor

    def run(self, task_id: str) -> None:
        """运行解析、清洗、分块、向量化、入库流水线。"""
        task = self.store.get_task(task_id)
        if task is None:
            raise ValueError(f"任务不存在: {task_id}")
        document = self.store.get_document(task.document_id)
        if document is None:
            failed = task.mark_failed("文档不存在")
            self.store.save_task(failed)
            return

        try:
            self.store.save_task(task.mark_running(BuildStep.PARSING, 20))
            pdf_path = Path(document.file_path)
            blocks = self.parser.parse_pdf(pdf_path)
            if self.post_processor is not None:
                blocks = self.post_processor.process(pdf_path, blocks)

            self.store.save_task(task.mark_running(BuildStep.CLEANING, 45))
            blocks = clean_blocks(blocks)

            self.store.save_task(task.mark_running(BuildStep.CHUNKING, 60))
            chunks = build_chunks(document.document_id, blocks)
            self.store.save_chunks(chunks)

            self.store.save_task(task.mark_running(BuildStep.EMBEDDING, 75))
            self.embedder.embed_texts([chunk.text for chunk in chunks])

            self.store.save_task(task.mark_running(BuildStep.UPSERTING, 90))
            self.vector_store.upsert_chunks(chunks, self.embedder)

            self.store.save_task(task.mark_completed())
            self.store.save_document(document.model_copy(update={"status": DocumentStatus.INDEXED}))
        except Exception as exc:
            self.store.save_task(task.mark_failed(str(exc)))
            self.store.save_document(document.model_copy(update={"status": DocumentStatus.FAILED}))
