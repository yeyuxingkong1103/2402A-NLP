from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Table:
    source: str
    page: int | None
    headers: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]

    def to_text(self) -> str:
        header = " | ".join(self.headers)
        rows = [" | ".join(row) for row in self.rows]
        return "\n".join([header, *rows])


def extract_tables(path: str | Path) -> list[Table]:
    try:
        import pdfplumber
    except ImportError as exc:
        raise RuntimeError("请先安装 pdfplumber：pip install pdfplumber") from exc

    tables: list[Table] = []
    with pdfplumber.open(str(path)) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            for raw_table in page.extract_tables() or []:
                cleaned = [tuple((cell or "").strip() for cell in row) for row in raw_table if row]
                if len(cleaned) >= 2:
                    tables.append(Table(str(path), page_number, cleaned[0], tuple(cleaned[1:])))
    return tables


def table_to_documents(tables: Iterable[Table]):
    """Convert extracted tables to the common Document shape."""
    from rag_core import Document
    return [Document(table.to_text(), table.source, table.page, {"content_type": "table"}) for table in tables]
