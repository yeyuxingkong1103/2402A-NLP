"""工单3：PDF 表格解析和表格问答索引。"""

from __future__ import annotations

import csv
import io
from pathlib import Path


def table_to_markdown(table: list[list[str | None]]) -> str:
    rows = [[(cell or "").strip().replace("\n", " ") for cell in row] for row in table]
    rows = [row for row in rows if any(row)]
    if not rows:
        return ""
    width = max(map(len, rows))
    rows = [row + [""] * (width - len(row)) for row in rows]
    return "\n".join(["| " + " | ".join(rows[0]) + " |", "| " + " | ".join("---" for _ in rows[0]) + " |", *["| " + " | ".join(row) + " |" for row in rows[1:]]])


def extract_tables(pdf_path: str, output_dir: str = "tables") -> list[dict]:
    import pdfplumber

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    results = []
    with pdfplumber.open(pdf_path) as document:
        for page_number, page in enumerate(document.pages, 1):
            for table_index, table in enumerate(page.extract_tables() or [], 1):
                markdown = table_to_markdown(table)
                if not markdown:
                    continue
                row = {"source": pdf_path, "page": page_number, "table_index": table_index, "text": markdown, "content_type": "table"}
                results.append(row)
                (target / f"page-{page_number:04d}-table-{table_index:02d}.md").write_text(markdown, encoding="utf-8")
    return results


def rows_to_csv(table: list[list[str | None]]) -> str:
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerows(table)
    return stream.getvalue()
