"""Work order 03: preserve PDF tables as searchable structured text."""

from __future__ import annotations

import csv
import io
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import Chunk, make_chunks, normalize_text


@dataclass
class TableRecord:
    source: str
    page: int | None
    table_index: int
    headers: list[str]
    rows: list[list[str]]

    def as_markdown(self) -> str:
        headers = self.headers or [f"col_{i + 1}" for i in range(len(self.rows[0]) if self.rows else 1)]
        width = len(headers)
        rows = [row[:width] + [""] * max(0, width - len(row)) for row in self.rows]
        lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
        lines.extend("| " + " | ".join(row) + " |" for row in rows)
        return "\n".join(lines)


def normalize_table(raw_rows: Iterable[Iterable[Any]], source: str, page: int | None, table_index: int) -> TableRecord:
    rows = [[normalize_text(str(cell or "")) for cell in row] for row in raw_rows]
    rows = [row for row in rows if any(row)]
    headers = rows[0] if rows else []
    return TableRecord(source, page, table_index, headers, rows[1:])


def extract_tables(pdf_path: str) -> list[TableRecord]:
    try:
        import pdfplumber  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Install pdfplumber or replace extract_tables with the project parser") from exc
    tables = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages, 1):
            for table_index, rows in enumerate(page.extract_tables() or []):
                tables.append(normalize_table(rows, pdf_path, page_number, table_index))
    return tables


def table_documents(tables: Iterable[TableRecord]) -> list[dict[str, Any]]:
    return [
        {
            "source": table.source,
            "page": table.page,
            "text": f"Table {table.table_index}\n{table.as_markdown()}",
            "metadata": {"content_type": "table", "table_index": table.table_index},
        }
        for table in tables
    ]


def table_aware_chunks(text_pages: list[dict[str, Any]], tables: Iterable[TableRecord]) -> list[Chunk]:
    documents = list(text_pages) + table_documents(tables)
    return make_chunks(documents, chunk_size=700, overlap=100)


def rows_to_csv(table: TableRecord) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(table.headers)
    writer.writerows(table.rows)
    return buffer.getvalue()
