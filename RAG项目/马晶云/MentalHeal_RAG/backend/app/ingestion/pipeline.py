import json
import logging
from pathlib import Path

from .config import IngestionConfig
from .mineru_client import MineruClient
from .schemas import ExtractedPage, ParsedDocument, TextChunk
from .text_processing import chunk_text, document_id

logger = logging.getLogger(__name__)


class PdfIngestionPipeline:
    def __init__(self, config: IngestionConfig) -> None:
        self.config = config
        self.mineru_client = MineruClient(
            api_key=config.mineru_api_key,
            base_url=config.mineru_base_url,
            api_version=config.mineru_api_version,
            model_version=config.mineru_model_version,
            poll_seconds=config.mineru_poll_seconds,
            timeout_seconds=config.mineru_timeout_seconds,
            is_ocr=config.mineru_is_ocr,
        )

    def process_file(self, path: Path) -> ParsedDocument:
        logger.info("Start MinerU-only PDF ingestion: %s", path.name)
        doc_id = document_id(path)
        image_dir = self.config.processed_dir / "mineru_images" / doc_id
        mineru_by_page = self.mineru_client.parse(path, image_dir)
        pages = self._build_pages(mineru_by_page)
        chunks = self._build_chunks(path, pages)
        result = ParsedDocument(
            document_id=doc_id,
            source_path=str(path),
            title=path.stem,
            page_count=len(pages),
            pages=pages,
            chunks=chunks,
            engines=["mineru"],
        )
        self._write_result(result)
        return result

    def process_directory(self) -> list[ParsedDocument]:
        self.config.raw_dir.mkdir(parents=True, exist_ok=True)
        self.config.processed_dir.mkdir(parents=True, exist_ok=True)
        self.config.failed_dir.mkdir(parents=True, exist_ok=True)
        results: list[ParsedDocument] = []
        for path in sorted(self.config.raw_dir.glob("*.pdf")):
            output = self.config.processed_dir / f"{document_id(path)}.json"
            if output.exists():
                logger.info("Skip already processed PDF: %s", path.name)
                continue
            try:
                results.append(self.process_file(path))
            except Exception as error:
                logger.exception("MinerU-only PDF ingestion failed: %s", path.name)
                self._write_failure(path, error)
        return results

    @staticmethod
    def _build_pages(mineru_by_page: dict[int, str]) -> list[ExtractedPage]:
        if not mineru_by_page:
            return []
        page_count = max(mineru_by_page)
        return [
            ExtractedPage(page_number=page_number, mineru_text=mineru_by_page.get(page_number, ""))
            for page_number in range(1, page_count + 1)
        ]

    def _build_chunks(self, path: Path, pages: list[ExtractedPage]) -> list[TextChunk]:
        chunks: list[TextChunk] = []
        doc_id = document_id(path)
        for page in pages:
            for index, text in enumerate(
                chunk_text(page.mineru_text, self.config.chunk_size, self.config.chunk_overlap), start=1
            ):
                chunks.append(
                    TextChunk(
                        chunk_id=f"{doc_id}-{page.page_number}-{index}",
                        page_start=page.page_number,
                        page_end=page.page_number,
                        text=text,
                        source=str(path),
                        metadata={"title": path.stem, "page": page.page_number},
                    )
                )
        return chunks

    def _write_result(self, result: ParsedDocument) -> None:
        target = self.config.processed_dir / f"{result.document_id}.json"
        target.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    def _write_failure(self, path: Path, error: Exception) -> None:
        target = self.config.failed_dir / f"{path.stem}.error.json"
        payload = {"source_path": str(path), "error_type": type(error).__name__, "error": str(error)}
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
