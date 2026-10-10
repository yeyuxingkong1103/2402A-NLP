# -*- coding: utf-8 -*-
"""PDF 文档解析模块（图像内容解析及检索优化版）
工单编号: 人工智能 NLP-RAG-图像内容解析及检索优化

本模块在 02 工单「版式清洗」基础上，进一步把表格解析从 `pdf_parser` 中解耦，
交由 `src.table_parser` 做结构化处理，形成：

    版面文本（版式清洗）  +  结构化表格（表格解析）  →  页面文档对象

相较基线（表格被拍平成普通文本行）：
1. 表格不再作为文本参与阅读顺序拼接，而是独立的结构化对象，保留行列关系；
2. 每张表携带表题（caption）与来源文档（source），便于多文档溯源；
3. 支持跨页表格的表题继承：若某页表格无表题，则继承上一页最近表题。
"""
from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from src import config
from src.table_parser import TableBlock, parse_tables

logger = logging.getLogger(__name__)

# 页眉页脚候选行：过短且为纯页码 / 罗马数字的行
_PAGE_NO_RE = re.compile(r"^[\s\-—]*[0-9]{1,4}[\s\-—]*$|^[\s\-—]*[IVXivx]{1,6}[\s\-—]*$")
# 章节标题：第X章 / 第X节 / 一、二、 / （一）（二） / 1.1 / 1.1.1
_HEADING_RE = re.compile(
    r"^\s*(?:第[一二三四五六七八九十百零〇\d]+[章节]\s*[^\n]{0,40}"
    r"|[一二三四五六七八九十]+、[^\n]{0,40}"
    r"|（[一二三四五六七八九十]+）[^\n]{0,40}"
    r"|\d+(?:\.\d+){1,3}\s+[^\n]{0,40})\s*$"
)


@dataclass
class PageDoc:
    """一页 PDF 的解析结果。"""

    page_no: int
    text: str = ""
    tables: List[TableBlock] = field(default_factory=list)
    headings: List[str] = field(default_factory=list)
    source: str = ""                      # 所属文档文件名（多文档溯源）

    @property
    def content(self) -> str:
        parts = [self.text.strip()]
        for t in self.tables:
            parts.append(f"[表格]\n{t.text}")
        return "\n".join(p for p in parts if p)


# ---------------- 版式清洗 ----------------------------------------------------
def _normalize_lines(text: str) -> List[str]:
    """按行拆分并规整空白。"""
    lines = []
    for raw in (text or "").splitlines():
        s = re.sub(r"[ \t\u3000]+", " ", raw).strip()
        lines.append(s)
    return lines


def _merge_wrapped_lines(lines: List[str]) -> List[str]:
    """把 PDF 换行造成的断句合并为完整行。

    规则：上一行不以句末标点结尾且当前行不是标题 / 表格行时，判定为同一段。
    """
    merged: List[str] = []
    for s in lines:
        if not s:
            merged.append("")
            continue
        if (
            merged
            and merged[-1]
            and not merged[-1].endswith(("。", "；", "：", "！", "？", "%", ")", "）", "】"))
            and not s.startswith(("|", "第", "一、", "二、", "三、", "（"))
            and not _HEADING_RE.match(s)
            and len(merged[-1]) > 12
        ):
            merged[-1] = merged[-1] + s
        else:
            merged.append(s)
    return merged


def detect_boilerplate(pages: List[PageDoc], ratio: float = 0.25) -> set:
    """统计跨页重复行，识别页眉 / 页脚等“模板文字”。"""
    counter: Counter = Counter()
    for pg in pages:
        lines = [s for s in pg.text.splitlines() if s.strip()]
        if not lines:
            continue
        for s in set(lines[:3] + lines[-3:]):
            counter[s.strip()] += 1
    threshold = max(3, int(len(pages) * ratio))
    return {line for line, n in counter.items() if n >= threshold}


def _clean_page_text(text: str, boiler: set) -> str:
    """剔除模板文字与纯页码行，并合并断句。"""
    lines = _normalize_lines(text)
    kept = []
    for s in lines:
        if not s:
            kept.append("")
            continue
        if s in boiler or _PAGE_NO_RE.match(s):
            continue
        kept.append(s)
    merged = _merge_wrapped_lines(kept)
    out = "\n".join(x for x in merged)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def _extract_headings(text: str) -> List[str]:
    return [s for s in text.splitlines() if s and _HEADING_RE.match(s)]


# ---------------- 主入口 ------------------------------------------------------
def parse_pdf(pdf_path, with_tables: bool = True,
              clean_boilerplate: bool = True,
              use_table_parser: bool | None = None) -> List[PageDoc]:
    """解析单个 PDF，返回按页组织的文档列表。

    Args:
        pdf_path: PDF 路径。
        with_tables: 是否解析表格。
        clean_boilerplate: 是否做页眉页脚清洗。
        use_table_parser: 是否启用结构化表格解析（默认取 config）。

    Raises:
        FileNotFoundError: 文件不存在。
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF 文件不存在: {path}")

    use_table_parser = config.USE_TABLE_PARSER if use_table_parser is None else use_table_parser

    try:
        import pymupdf
    except ImportError:
        import fitz as pymupdf

    pages: List[PageDoc] = []
    with pymupdf.open(str(path)) as doc:
        for i in range(doc.page_count):
            page = doc.load_page(i)
            try:
                text = page.get_text("text", sort=True) or ""
            except Exception:
                text = ""
            pages.append(PageDoc(page_no=i + 1, text=text, source=path.name))

    # ---- 结构化表格解析（本工单新增） ----
    if with_tables and use_table_parser:
        blocks = parse_tables(path, source_name=path.name)
        by_page: dict[int, List[TableBlock]] = {}
        for b in blocks:
            by_page.setdefault(b.page, []).append(b)
        last_caption = ""
        for pg in pages:                      # 按页顺序继承表题（跨页表格）
            for b in by_page.get(pg.page_no, []):
                if b.caption:
                    last_caption = b.caption
                elif last_caption and not b.caption:
                    b.caption = last_caption
            pg.tables = by_page.get(pg.page_no, [])

    # ---- 版式清洗 ----
    if clean_boilerplate and pages:
        boiler = detect_boilerplate(pages)
        logger.info("[%s] 识别模板文字 %d 条", path.name, len(boiler))
        for pg in pages:
            pg.text = _clean_page_text(pg.text, boiler)
            pg.headings = _extract_headings(pg.text)

    return pages


def parse_all(pdf_paths=None, **kwargs) -> List[PageDoc]:
    """解析全部知识库文档（多文档支持）。"""
    paths = pdf_paths or config.PDF_PATHS
    out: List[PageDoc] = []
    for p in paths:
        out.extend(parse_pdf(p, **kwargs))
    return out


def parse_to_text(pdf_path) -> str:
    """解析 PDF 并返回全文纯文本（便于人工核对答案）。"""
    pages = parse_pdf(pdf_path, with_tables=True)
    return "\n\n".join(f"—— {p.source} 第 {p.page_no} 页 ——\n{p.content}" for p in pages)


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    logging.basicConfig(level=logging.INFO)

    pages = parse_all()
    print(f"总页数: {len(pages)}")
    print(f"总字符: {sum(len(p.content) for p in pages)}")
    print(f"含表格页数: {sum(1 for p in pages if p.tables)}")
    print(f"表格总数: {sum(len(p.tables) for p in pages)}")
    for p in pages[:2]:
        print("=" * 40)
        print(p.source, p.page_no)
        print(p.content[:400])