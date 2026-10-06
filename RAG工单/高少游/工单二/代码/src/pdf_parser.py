# -*- coding: utf-8 -*-
"""PDF 文档解析模块（优化版）
工单编号: 人工智能 NLP-RAG-基于 PDF 文档的问答系统优化

本模块对应优化方案中的【优化点 1：PDF 解析处理】。相较基线（直接
page.get_text 后原样切片），做了如下增强：

1. 版式清洗：自动识别并剔除跨页重复的页眉 / 页脚（如“武汉兴图新科电子股份
   有限公司 招股意向书”、页码等），避免其稀释检索语义、干扰关键词匹配；
2. 表格结构化：页内表格渲染为 Markdown（保留行列关系），并单独作为“原子块”
   参与建库，避免财务数据在切片时被截断丢失；
3. 行合并与空白规整：修复 PDF 换行造成的断句，压缩多余空白，提升切片质量；
4. 章节标题识别：抽取“第X节 / 一、/（一）”等层级标题，为结构化分块提供锚点；
5. 容错：单页文字 / 表格解析异常时降级跳过，不影响整体流程。
"""
from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

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
    tables: List[str] = field(default_factory=list)
    headings: List[str] = field(default_factory=list)

    @property
    def content(self) -> str:
        parts = [self.text.strip()]
        for t in self.tables:
            parts.append(f"[表格]\n{t}")
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
    """统计跨页重复行，识别页眉 / 页脚等“模板文字”。

    Args:
        pages: 解析出的页列表。
        ratio: 出现在超过该比例页面上的行，判定为模板文字。

    Returns:
        需要剔除的文本行集合。
    """
    counter: Counter = Counter()
    for pg in pages:
        lines = [s for s in pg.text.splitlines() if s.strip()]
        if not lines:
            continue
        # 仅统计每页首尾各 3 行（页眉页脚高发区）
        for s in set(lines[:3] + lines[-3:]):
            counter[s.strip()] += 1
    threshold = max(3, int(len(pages) * ratio))
    boiler = {line for line, n in counter.items() if n >= threshold}
    return boiler


def _clean_page_text(text: str, boiler: set) -> str:
    """剔除模板文字与纯页码行，并合并断句。"""
    lines = _normalize_lines(text)
    kept = []
    for s in lines:
        if not s:
            kept.append("")
            continue
        if s in boiler:
            continue
        if _PAGE_NO_RE.match(s):
            continue
        kept.append(s)
    merged = _merge_wrapped_lines(kept)
    out = "\n".join(x for x in merged)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def _render_table(table) -> str:
    """把 pymupdf 表格对象渲染为 Markdown 表格（保留行列关系）。"""
    try:
        if not table:
            return ""
        rows = table.extract()
        if not rows:
            return ""
        lines = []
        for row in rows:
            cells = [str(c).replace("\n", " ").strip() if c else "" for c in row]
            lines.append("| " + " | ".join(cells) + " |")
        if lines:
            n = max(1, len(rows[0]))
            lines.insert(1, "|" + "---|" * n)
        return "\n".join(lines)
    except Exception as exc:
        logger.debug("table render failed: %s", exc)
        return ""


def _extract_headings(text: str) -> List[str]:
    return [s for s in text.splitlines() if s and _HEADING_RE.match(s)]


# ---------------- 主入口 ------------------------------------------------------
def parse_pdf(pdf_path: str | Path, with_tables: bool = True,
              clean_boilerplate: bool = True) -> List[PageDoc]:
    """解析 PDF，返回按页组织的文档列表（已做版式清洗）。

    Raises:
        FileNotFoundError: 文件不存在。
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF 文件不存在: {path}")

    try:
        import pymupdf as fitz
    except ImportError:
        import fitz

    pages: List[PageDoc] = []
    with fitz.open(str(path)) as doc:
        for i in range(doc.page_count):
            page = doc.load_page(i)
            try:
                text = page.get_text("text", sort=True) or ""
            except Exception:
                text = ""

            tables: List[str] = []
            if with_tables:
                try:
                    for tb in (page.find_tables() or []):
                        rendered = _render_table(tb)
                        if rendered:
                            tables.append(rendered)
                except Exception as exc:
                    logger.debug("page %s table parse failed: %s", i + 1, exc)

            pages.append(PageDoc(page_no=i + 1, text=text, tables=tables))

    if clean_boilerplate and pages:
        boiler = detect_boilerplate(pages)
        logger.info("识别模板文字 %d 条", len(boiler))
        for pg in pages:
            pg.text = _clean_page_text(pg.text, boiler)
            pg.headings = _extract_headings(pg.text)

    return pages


def parse_to_text(pdf_path: str | Path) -> str:
    """解析 PDF 并返回全文纯文本（便于人工核对答案）。"""
    pages = parse_pdf(pdf_path, with_tables=True)
    return "\n\n".join(f"—— 第 {p.page_no} 页 ——\n{p.content}" for p in pages)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.config import PDF_PATH

    pages = parse_pdf(PDF_PATH)
    print(f"总页数: {len(pages)}")
    print(f"总字符: {sum(len(p.content) for p in pages)}")
    print(f"表格页数: {sum(1 for p in pages if p.tables)}")
    for p in pages[:2]:
        print("=" * 40)
        print(p.content[:600])