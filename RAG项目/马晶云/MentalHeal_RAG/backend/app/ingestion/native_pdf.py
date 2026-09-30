from pathlib import Path

import fitz

from .schemas import ExtractedPage
from .text_processing import normalize_text


class NativePdfParser:
    name = "pymupdf"

    def parse(self, path: Path) -> list[ExtractedPage]:
        pages: list[ExtractedPage] = []
        with fitz.open(path) as document:
            for index, page in enumerate(document):
                pages.append(
                    ExtractedPage(
                        page_number=index + 1,
                        native_text=normalize_text(page.get_text("text")),
                    )
                )
        return pages
