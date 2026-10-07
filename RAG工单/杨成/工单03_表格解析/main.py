"""Markdown、文本及可选 PDF 表格抽取和索引适配。"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
from typing import Any, Iterable

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.models import TableRecord


def _cell_text(value: object) -> str:
    return "" if value is None else str(value).strip()
def _cells(line: str) -> list[str]:
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [cell.strip() for cell in stripped.split("|")]


def _is_separator(line: str) -> bool:
    cells = _cells(line)
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell.replace(" ", "")) for cell in cells)


def parse_table(table: str) -> tuple[list[str], list[list[str]]]:
    """识别 Markdown、TSV 或连续空格分隔的表格。"""
    lines = [line.rstrip("\r") for line in table.splitlines() if line.strip()]
    if len(lines) < 2:
        raise ValueError("输入不是有效的表格：至少需要表头和一行数据")
    if "|" in lines[0] and _is_separator(lines[1]):
        headers = _cells(lines[0])
        rows = [_cells(line) for line in lines[2:] if "|" in line]
    else:
        splitter = (lambda line: [cell.strip() for cell in line.split("\t")]) if "\t" in lines[0] else (lambda line: re.split(r"\s{2,}", line.strip()))
        headers = splitter(lines[0])
        rows = [splitter(line) for line in lines[1:]]
    if len(headers) < 2:
        raise ValueError("表格至少需要两列")
    if any(len(row) != len(headers) for row in rows):
        raise ValueError("表格行列数与表头不一致")
    return headers, rows


def markdown_table_to_text(table: str) -> str:
    """将表头与每行拼接，确保检索命中行时携带字段语义。"""
    headers, rows = parse_table(table)
    if not rows:
        return " | ".join(headers)
    return "\n".join(
        "；".join(f"{header}: {value}" for header, value in zip(headers, row))
        for row in rows
    )


def index_table(source: str, page: int, table: str, index: Any | None = None) -> TableRecord:
    """创建表格记录；传入统一索引时同时写入该索引。"""
    headers, rows = parse_table(table)
    record = TableRecord(
        source=source,
        page=page,
        content=markdown_table_to_text(table),
        metadata={"rows": len(rows), "columns": len(headers), "headers": headers, "raw_table": table},
    )
    if index is not None:
        if hasattr(index, "add_table"):
            index.add_table(record)
        elif hasattr(index, "add_text"):
            try:
                index.add_text(record.source, record.page, record.content, metadata={**record.metadata, "table_id": record.table_id})
            except TypeError:
                index.add_text(record.source, record.page, record.content)
        else:
            raise TypeError("index 必须支持 add_table 或 add_text")
    return record


def extract_tables(text: str, source: str = "", page: int = 1) -> list[TableRecord]:
    """从文本中识别连续 Markdown 表格并生成记录。"""
    lines = text.splitlines()
    records: list[TableRecord] = []
    position = 0
    while position < len(lines) - 1:
        if "|" not in lines[position] or not _is_separator(lines[position + 1]):
            position += 1
            continue
        end = position + 2
        while end < len(lines) and "|" in lines[end] and lines[end].strip():
            end += 1
        records.append(index_table(source, page, "\n".join(lines[position:end])))
        position = end
    return records


def extract_pdf_tables(path: str | Path, pages: Iterable[int] | None = None) -> list[TableRecord]:
    """优先使用 camelot，其次 pdfplumber；缺少依赖时给出明确提示。"""
    pdf_path = Path(path)
    selected_pages = set(pages or [])
    try:
        import camelot  # type: ignore
    except ImportError:
        camelot = None
    if camelot is not None:
        try:
            page_spec = ",".join(map(str, sorted(selected_pages))) if selected_pages else "1-end"
            tables = camelot.read_pdf(str(pdf_path), pages=page_spec)
            if tables:
                records = []
                for table in tables:
                    values = table.df.values
                    rows = values.tolist() if hasattr(values, "tolist") else values
                    if not rows or not rows[0]:
                        continue
                    page = int(getattr(table, "page", 1))
                    if selected_pages and page not in selected_pages:
                        continue
                    headers, data = rows[0], rows[1:]
                    markdown = "| " + " | ".join(_cell_text(value) for value in headers) + " |\n| " + " | ".join("---" for _ in headers) + " |\n"
                    markdown += "\n".join("| " + " | ".join(_cell_text(value) for value in row) + " |" for row in data)
                    records.append(index_table(str(pdf_path), page, markdown))
                if records:
                    return records
        except Exception:
            pass
    try:
        import pdfplumber  # type: ignore
    except ImportError as exc:
        raise RuntimeError("PDF 表格抽取需要安装 camelot 或 pdfplumber；也可先提供结构化 Markdown 表格") from exc
    records = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page_number, pdf_page in enumerate(pdf.pages, 1):
            if selected_pages and page_number not in selected_pages:
                continue
            for extracted in pdf_page.extract_tables() or []:
                if not extracted or not extracted[0]:
                    continue
                headers = [str(value or "").strip() for value in extracted[0]]
                rows = [[str(value or "").strip() for value in row] for row in extracted[1:]]
                markdown = "| " + " | ".join(headers) + " |\n| " + " | ".join("---" for _ in headers) + " |\n"
                markdown += "\n".join("| " + " | ".join(row) + " |" for row in rows)
                records.append(index_table(str(pdf_path), page_number, markdown))
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="抽取并格式化 PDF/Markdown 表格")
    parser.add_argument("path", type=Path)
    parser.add_argument("--page", type=int, default=None)
    args = parser.parse_args(argv)
    try:
        if args.path.suffix.lower() == ".pdf":
            records = extract_pdf_tables(args.path, [args.page] if args.page else None)
        else:
            raw = args.path.read_text(encoding="utf-8")
            headers, rows = parse_table(raw)
            records = [index_table(str(args.path), 1, raw)]
    except (OSError, UnicodeError, ValueError, RuntimeError) as exc:
        print(f"错误：{exc}", file=__import__("sys").stderr)
        return 1
    for record in records:
        print(f"[{record.source} 第{record.page}页]\n{record.content}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
