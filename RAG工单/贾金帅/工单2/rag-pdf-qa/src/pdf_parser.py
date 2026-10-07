"""
PDF 解析模块
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统

职责：把 PDF 还原成「按页组织的干净文本 + 表格」。对应工单产出物里的
「PDF解析模块：支持附件PDF文档的解析与提取功能」。

两个要点：
1) 招股说明书每页都有重复页眉（公司名 + 招股意向书）和形如 1-1-30 的页码，
   这些是纯噪声，必须在入索引前剔除，否则会被 BM25 当成高频关键词命中。
2) 财务报表类内容用 pdfplumber 抽成 Markdown 表格。纯文本行会把
   「2019年1-6月 / 2018年度 / 2017年度」这类列头与数值切散，语义断裂，
   问答时模型无法对齐行列。表格单独成块，检索命中后可直接读表。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .config import WORK_ORDER_NO  # noqa: F401  (工单编号，模块标识)

# ------------------------------------------------------------------ 噪声规则

# 页眉：公司名 + 招股意向书（中间可能有大量空白）
_HEADER_RE = re.compile(r"^\s*武汉兴图新科电子股份有限公司\s*招股意向书\s*$")
# 反向页眉（左右位置互换时）
_HEADER_RE2 = re.compile(r"^\s*招股意向书\s*武汉兴图新科电子股份有限公司\s*$")
# 页码：1-1-30 / 1-1-1 形式，或纯数字行
_PAGENO_RE = re.compile(r"^\s*\d+-\d+-\d+\s*$")
_BARE_PAGENO_RE = re.compile(r"^\s*第?\s*\d{1,3}\s*页?\s*$")
# 大量连续下划线/点线（表格占位或分页符残留）
_FILLER_RE = re.compile(r"^[\s_\-\.·。]{6,}$")


@dataclass
class PageText:
    """单页解析结果。"""

    page: int
    text: str
    tables: list[str] = field(default_factory=list)


@dataclass
class ParsedDoc:
    """整份文档的解析结果。"""

    source: str
    total_pages: int
    pages: list[PageText]

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages if p.text)


def _clean_line(line: str) -> str:
    """去掉行内多余空白，保留中文原样。"""
    line = line.replace("\u3000", " ")  # 全角空格
    line = re.sub(r"[ \t]{2,}", " ", line)
    return line.strip()


def _is_noise(line: str) -> bool:
    if not line:
        return True
    if _HEADER_RE.match(line) or _HEADER_RE2.match(line):
        return True
    if _PAGENO_RE.match(line) or _BARE_PAGENO_RE.match(line):
        return True
    if _FILLER_RE.match(line):
        return True
    return False


# 行内标题：Word 转 PDF 后，标题常与上一段被压进同一行，例如
#   "……数字传输网络。第二节 概览 本概览仅对招股意向书全文做扼要提示。"
# 行首匹配的标题识别因此全部失效 → 后续分块会把两个大节混进一块，
# 块向量被两个主题拉扯，检索时既不像上面也不像下面（实测「注册资本」问题
# 因为这条，正确答案所在块被挤出上下文）。
# 修法：在紧跟在句末标点之后的层级标题前强制断行。
_INLINE_HEADING_RE = re.compile(
    r"(?<=[。！？；])"
    r"(?=(?:第[一二三四五六七八九十百]+[节章]|[一二三四五六七八九十]+、|（[一二三四五六七八九十]+）))"
)


def split_inline_headings(text: str) -> str:
    """把「段尾标点 + 标题」拆成两行，让标题重新落到行首。"""
    return _INLINE_HEADING_RE.sub("\n", text)


def clean_page_text(raw: str, split_inline: bool = True) -> str:
    """
    清洗单页文本：逐行去噪 + 行内标题断行。

    split_inline=False 时**跳过行内标题断行**——这是工单02 消融实验的开关：
    朴素基线直接取页文本，不做这一步，用来量化「解析层优化」单独的贡献。
    """
    lines = [_clean_line(ln) for ln in raw.splitlines()]
    lines = [ln for ln in lines if not _is_noise(ln)]
    joined = "\n".join(lines).strip()
    return split_inline_headings(joined) if split_inline else joined


# ------------------------------------------------------------------ 表格抽取

# 只要数值列 ≥3 行，就认为该页值得跑一次表格抽取
_NUM_RE = re.compile(r"[\d,]+\.?\d*")


def _looks_tabular(text: str) -> bool:
    """快速启发式：判断某页是否可能含表格，避免对 548 页全量跑 pdfplumber。"""
    numeric_lines = 0
    for ln in text.splitlines():
        nums = _NUM_RE.findall(ln)
        # 一行里出现 3 个以上数字，且不是纯页码，视为疑似表格行
        if len(nums) >= 3:
            numeric_lines += 1
    return numeric_lines >= 3


def _table_to_markdown(rows: list[list[str | None]]) -> str:
    """把 pdfplumber 的表格二维数组转成 Markdown，保留行列结构。"""
    clean: list[list[str]] = []
    for row in rows:
        cells = [(_clean_line(c) if c else "") for c in row]
        if any(cells):
            clean.append(cells)
    if len(clean) < 2:
        return ""
    width = max(len(r) for r in clean)
    clean = [r + [""] * (width - len(r)) for r in clean]

    head = clean[0]
    body = clean[1:]
    md = ["| " + " | ".join(head) + " |", "|" + "---|" * width]
    for r in body:
        md.append("| " + " | ".join(r) + " |")
    return "\n".join(md)


def extract_tables_batch(pdf_path: Path, page_indexes: list[int]) -> dict[int, list[str]]:
    """
    批量抽取指定页（0-based）的表格。

    **只打开一次 PDF**：pdfplumber.open 要解析整个 xref 表，对 11MB / 548 页的文档
    约 1~2 秒。若按「每页单独 open 一次」写，就是 O(n²)，实测会从 2 分钟退化到 3 分钟以上。
    """
    import pdfplumber  # 局部导入：不跑表格时无需付出导入开销

    out: dict[int, list[str]] = {}
    if not page_indexes:
        return out

    with pdfplumber.open(str(pdf_path)) as pdf:
        pages = pdf.pages
        for idx in page_indexes:
            if idx < 0 or idx >= len(pages):
                continue
            try:
                tables = pages[idx].extract_tables() or []
            except Exception:
                tables = []
            mds = [md for md in (_table_to_markdown(t) for t in tables) if md]
            if mds:
                out[idx] = mds
    return out


def parse_pdf(
    pdf_path: Path,
    with_tables: bool = True,
    split_inline: bool = True,
) -> ParsedDoc:
    """
    解析 PDF。返回按页组织的文本；with_tables=True 时对疑似表格页补抽表格。

    split_inline=False 关闭「行内标题断行」（工单02 消融实验用，见 clean_page_text）。

    注意：为了对 548 页保持可接受的耗时（本机实测全量 pdfplumber 约 6~9 分钟），
    这里先用 `_looks_tabular` 做启发式筛选，只对疑似页打开 pdfplumber。
    """
    from pypdf import PdfReader

    pdf_path = Path(pdf_path)
    reader = PdfReader(str(pdf_path))
    pages: list[PageText] = []
    suspect_pages: list[int] = []

    for i, page in enumerate(reader.pages):
        try:
            raw = page.extract_text() or ""
        except Exception:  # 单页解析失败不应中断整份文档（工单「容错机制」要求）
            raw = ""
        text = clean_page_text(raw, split_inline=split_inline)
        pages.append(PageText(page=i + 1, text=text))
        if with_tables and _looks_tabular(text):
            suspect_pages.append(i)

    if with_tables and suspect_pages:
        for idx, mds in extract_tables_batch(pdf_path, suspect_pages).items():
            pages[idx].tables = mds

    return ParsedDoc(source=pdf_path.name, total_pages=len(pages), pages=pages)


def parse_pdf_text_only(pdf_path: Path) -> ParsedDoc:
    """只做文字解析（不抽表格），用于快速干跑与分块回归。"""
    return parse_pdf(pdf_path, with_tables=False)
