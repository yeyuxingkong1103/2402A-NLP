# -*- coding: utf-8 -*-
"""T8 只读 PDF 取证工具（引用可回溯核验的地基）。

工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化

为什么需要它：验收 5 要求「引用页码必须能在对应 PDF 真实页找到证据原文」。
若直接信产品的 ``page`` 字段，测试就退化成自证；因此这里用 ``pymupdf`` 独立打开
原始 PDF 取页文本，作为**外部基准**（与产品解析链路完全独立）。

口径（``设计/需求分析.md`` §4.4）：
    * ``d[i]`` = 物理页 ``i+1``（**1-based 物理页是全仓库唯一页码口径**）；
    * 页脚 ``1-1-128``（PDF1）与页眉裸数字 ``128``（PDF2）都是 **0-based 索引**，
      禁止用于引用，也不得用其反推页码；
    * PDF2 的裸数字位于页眉块**第 2 行**：清洗必须「位置 + 值」双判据。

本模块全部只读（``doc.close()`` 收尾），不写任何缓存文件。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator

from . import paths


def _open(path: Path | str):
    """打开 PDF（延迟导入 pymupdf：未装时给出明确错误，不静默返回空）。"""
    import pymupdf  # noqa: PLC0415 —— 延迟导入，避免收集阶段硬依赖

    return pymupdf.open(str(path))


@lru_cache(maxsize=8)
def page_count(path_str: str) -> int:
    """物理页数（1-based 上限）。"""
    doc = _open(path_str)
    try:
        return int(doc.page_count)
    finally:
        doc.close()


@lru_cache(maxsize=4096)
def page_text(path_str: str, page: int) -> str:
    """取第 ``page`` 页（1-based）的 ``get_text("text")`` 原文。

    非法页码 → ``ValueError``（显式失败，便于把「引用越界」判成缺陷而不是空串）。
    """
    total = page_count(path_str)
    if not isinstance(page, int) or page < 1 or page > total:
        raise ValueError(f"页码越界：page={page!r}，合法区间 1..{total}（{Path(path_str).name}）")
    doc = _open(path_str)
    try:
        return doc[page - 1].get_text("text")
    finally:
        doc.close()


@lru_cache(maxsize=512)
def table_count(path_str: str, page: int) -> int:
    """``page.find_tables()`` 命中的表格数（用于锚点对账：152 应为 0、153 应为 1、157 应为 2）。"""
    total = page_count(path_str)
    if not isinstance(page, int) or page < 1 or page > total:
        raise ValueError(f"页码越界：page={page!r}，合法区间 1..{total}")
    doc = _open(path_str)
    try:
        return len(doc[page - 1].find_tables().tables)
    finally:
        doc.close()


def lookup_factory() -> Any:
    """构造 ``page_text_lookup(file_name, page) -> str``（自动发现结果解析文件名）。

    返回的函数在文件不存在或页码越界时**抛异常**，保证断言侧不会拿到静默空串。
    """
    mapping = {p.name: str(p) for p in paths.discover_pdf_files()}

    def lookup(file_name: str, page: int) -> str:
        """按文件名取页文本；文件未知 → KeyError。"""
        if file_name not in mapping:
            raise KeyError(f"未知文件名 {file_name!r}；已发现：{sorted(mapping)}")
        return page_text(mapping[file_name], int(page))

    return lookup


def page_counts() -> dict[str, int]:
    """``{文件名: 页数}``（引用越界判定的基准）。"""
    return {p.name: page_count(str(p)) for p in paths.discover_pdf_files()}


def iter_pages(path: Path | str) -> Iterator[tuple[int, str]]:
    """逐页产出 ``(物理页, 文本)``（1-based）。"""
    doc = _open(path)
    try:
        for index in range(doc.page_count):
            yield index + 1, doc[index].get_text("text")
    finally:
        doc.close()


def printed_page_forms(physical_page: int) -> list[str]:
    """给定物理页，返回该页在 PDF 里**会出现但禁止用于引用**的数字形态。

    PDF1 页脚 = ``1-1-{物理页-1}``；PDF2 页眉第 2 行 = ``{物理页-1}``。
    两者都等于 0-based 索引，故它们与「正确的物理页码」恒差 1 —— 这正是 §4.4 的埋雷点。
    """
    return [str(int(physical_page) - 1), f"1-1-{int(physical_page) - 1}"]


def page_number_evidence(path: Path | str, page: int) -> dict[str, Any]:
    """取证「该物理页上印刷的数字」：页脚 ``1-1-N`` 与页眉第 2 行裸数字。

    返回 ``{"physical": page, "footer_printed": [...], "header_second_line": [...], "forbidden": [...]}``
    供离线用例断言「产品的引用页码不是这两类形态」。
    """
    text = page_text(str(path), page)
    lines = [ln.strip() for ln in text.splitlines()]
    footer = [ln for ln in lines if ln.startswith("1-1-")]
    header_second = [lines[1].strip()] if len(lines) > 1 else []
    return {
        "physical": int(page),
        "footer_printed": footer,
        "header_second_line": header_second,
        "forbidden": printed_page_forms(page),
        "matches_forbidden": any(
            line in printed_page_forms(page) for line in (footer + header_second)
        ),
    }
