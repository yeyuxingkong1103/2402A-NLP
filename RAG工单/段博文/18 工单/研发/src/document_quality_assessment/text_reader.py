# -*- coding: utf-8 -*-
"""统一文本读取：按文件类型提取纯文本，统一异常出口。"""
from __future__ import annotations

from pathlib import Path
from typing import Optional


class DocumentReadError(Exception):
    """文件损坏/无法读取时抛出，由上层捕获并标记 Corrupt_File。"""


def read_text(path: Path) -> str:
    """提取全文。无法解析或文件损坏时抛 DocumentReadError。"""
    ext = path.suffix.lower()
    try:
        if ext == ".pdf":
            return _read_pdf(path)
        if ext in (".docx", ".doc"):
            return _read_docx(path)
        if ext in (".md", ".txt"):
            return _read_plain(path)
    except DocumentReadError:
        raise
    except Exception as e:  # 任何底层异常统一包装
        raise DocumentReadError(f"{type(e).__name__}: {e}") from e
    raise DocumentReadError(f"不支持的文件类型: {ext}")


def _read_pdf(path: Path) -> str:
    import pymupdf  # 新版 API

    try:
        doc = pymupdf.open(path)
    except Exception as e:
        raise DocumentReadError(f"PDF 无法打开: {e}") from e
    try:
        parts = []
        for page in doc:
            parts.append(page.get_text("text"))
        return "".join(parts)
    finally:
        doc.close()


def _read_docx(path: Path) -> str:
    try:
        from docx import Document
    except ImportError as e:
        raise DocumentReadError("缺少 python-docx 依赖") from e
    doc = Document(str(path))
    paras = [p.text for p in doc.paragraphs]
    # 表格文本也纳入
    for tbl in doc.tables:
        for row in tbl.rows:
            paras.append("\t".join(cell.text for cell in row.cells))
    return "\n".join(paras)


def _read_plain(path: Path) -> str:
    raw = path.read_bytes()
    for enc in ("utf-8", "gb18030", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def open_pdf_pages(path: Path):
    """供 PDF 分类器使用：逐页产出 (页字符数)，不一次性加载到内存。

    返回 (page_count, page_char_counts)；打不开时抛 DocumentReadError。
    """
    page_count, page_chars, _ = read_pdf_detailed(path)
    return page_count, page_chars


def read_pdf_detailed(path: Path):
    """一次打开同时拿到 (页数, 逐页非空白字符数, 全文)，避免重复打开 PDF。"""
    import pymupdf

    try:
        doc = pymupdf.open(path)
    except Exception as e:
        raise DocumentReadError(f"PDF 无法打开: {e}") from e
    try:
        page_chars, parts = [], []
        for page in doc:
            t = page.get_text("text") or ""
            parts.append(t)
            page_chars.append(len("".join(t.split())))
        return doc.page_count, page_chars, "".join(parts)
    finally:
        doc.close()
