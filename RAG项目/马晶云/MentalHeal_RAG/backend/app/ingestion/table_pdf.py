from pathlib import Path

import pdfplumber

from .schemas import ExtractedPage
from .text_processing import normalize_text


class TablePdfParser:
    name = "pdfplumber"

    def parse(self, path: Path) -> dict[int, list[str]]:
        tables_by_page: dict[int, list[str]] = {}
        with pdfplumber.open(path) as document:
            for index, page in enumerate(document.pages, start=1):
                page_tables: list[str] = []
                for table in page.extract_tables() or []:
                    rows = [self._format_row(row) for row in table if row]
                    rows = [row for row in rows if row]
                    if rows:
                        page_tables.append("表格：\n" + "\n".join(rows))
                if page_tables:
                    tables_by_page[index] = page_tables
        return tables_by_page

    def apply(self, pages: list[ExtractedPage], tables_by_page: dict[int, list[str]]) -> None:
        for page in pages:
            page.tables.extend(tables_by_page.get(page.page_number, []))

    @staticmethod
    def _format_row(row: list[str | None]) -> str:
        cells = [normalize_text(cell or "") for cell in row]
        cells = [cell.replace("\n", " / ") for cell in cells]
        return " | ".join(cells).strip(" |")
