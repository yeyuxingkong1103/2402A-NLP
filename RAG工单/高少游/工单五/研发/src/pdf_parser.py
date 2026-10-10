# -*- coding: utf-8 -*-
"""PDF 解析：抽取正文文本与结构化表格。

工单编号: 人工智能 NLP-RAG-Query 理解优化任务

沿用 01~04 工单的「文本 + 表格」双通道解析思路：
    - 文本通道：按页抽取正文，保留页码用于溯源；
    - 表格通道：pdfplumber 抽取表格并渲染为「表头: 值」的键值对文本，
      便于「发行股数 / 法定代表人 / 军用领域收入」等字段型问题精准命中。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import pdfplumber

from src import config


@dataclass
class PageData:
    """单页解析结果。"""
    source: str            # 源文件名
    page: int              # 页码（1 起）
    text: str = ""         # 正文文本
    tables: List[str] = field(default_factory=list)   # 表格的键值对化文本


def _clean(text: str) -> str:
    text = text or ""
    text = text.replace("\u3000", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _table_to_kv(table: List[List[str | None]]) -> str:
    """把表格渲染为「表头: 值」形式的可检索文本。

    对每行，用第一列作为字段名，其余列以表头拼接，形成
    “军用领域收入: 6,464.51万元” 这类贴近自然语言的键值对，
    显著提升字段型问题的召回率。
    """
    rows = [[(c or "").strip() for c in row] for row in (table or [])]
    rows = [r for r in rows if any(c for c in r)]
    if len(rows) < config.TABLE_MIN_ROWS:
        return ""
    header = rows[0]
    if len(header) < config.TABLE_MIN_COLS and len(rows[0]) < 2:
        return ""
    lines: List[str] = []
    for row in rows[1:]:
        key = row[0]
        if not key:
            continue
        pairs = []
        for j, cell in enumerate(row[1:], start=1):
            if not cell:
                continue
            col = header[j] if j < len(header) else ""
            pairs.append(f"{col} {cell}".strip() if col else cell)
        if pairs:
            lines.append(f"{key}: " + "；".join(pairs))
    if not lines:
        # 表头本身也可能承载信息（如“项目 | 金额”）
        lines = [" | ".join(c for c in header if c)]
    return "\n".join(lines)


def parse_pdf(path: Path, max_pages: int | None = None,
              with_tables: bool = True) -> List[PageData]:
    """解析单个 PDF，返回逐页数据。"""
    pages: List[PageData] = []
    source = Path(path).name
    with pdfplumber.open(str(path)) as pdf:
        total = len(pdf.pages)
        limit = min(total, max_pages) if max_pages else total
        for i in range(limit):
            pg = pdf.pages[i]
            text = _clean(pg.extract_text() or "")
            tables: List[str] = []
            if with_tables:
                try:
                    for tbl in pg.extract_tables() or []:
                        kv = _table_to_kv(tbl)
                        if kv:
                            tables.append(kv)
                except Exception:
                    pass
            pages.append(PageData(source=source, page=i + 1, text=text, tables=tables))
    return pages


def parse_all(max_pages: int | None = None, with_tables: bool = True) -> List[PageData]:
    """解析配置中的全部 PDF。"""
    out: List[PageData] = []
    for p in config.PDF_PATHS:
        out.extend(parse_pdf(p, max_pages=max_pages, with_tables=with_tables))
    return out