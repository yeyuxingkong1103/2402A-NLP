"""
PDF 解析模块
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
工单编号：人工智能NLP-RAG-图像内容解析及检索优化

职责：把 PDF 还原成「按页组织的干净文本 + 结构化表格」。

三个要点：
1) 招股说明书每页都有重复页眉（公司名 + 招股(意向)书）和形如 3-21 的页码，
   这些是纯噪声，必须在入索引前剔除，否则会被 BM25 当成高频关键词命中。
   **工单03 起页眉规则按文档配置**（见 config.DOCS）—— 语料从单文档变成了两文档，
   写死一家公司名的正则会漏掉另一家的页眉。

2) 行内标题断行：Word 转 PDF 会把标题并进上一行，导致行首标题正则全废。

3) 表格抽取（工单03 核心）走 `table_parser`，不再用 pdfplumber 的
   extract_tables() 直接转 Markdown。原因见 src/table_parser.py 的模块注释：
   旧做法既漏表（数字密度启发式漏掉关联方表）又切错列（双线边框被当列）。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .config import WORK_ORDER_NOS  # noqa: F401  (工单编号，模块标识)
from .table_parser import Table, parse_pdf_tables  # noqa: F401

# ------------------------------------------------------------------ 噪声规则

# 默认页眉（未指定文档配置时的兜底）
_HEADER_RES_DEFAULT = [
    re.compile(r"^\s*武汉兴图新科电子股份有限公司\s*招股意向书\s*$"),
    re.compile(r"^\s*招股意向书\s*武汉兴图新科电子股份有限公司\s*$"),
    re.compile(r"^\s*武汉力源信息技术股份有限公司\s*招股说明书\s*$"),
    re.compile(r"^\s*招股说明书\s*武汉力源信息技术股份有限公司\s*$"),
]
# 页码：1-1-30 这种三段式（兴图），或 3-21 这种两段式（力源，形如 3-页数）
_PAGENO_RE = re.compile(r"^\s*\d+-\d+-\d+\s*$")
_PAGENO_RE2 = re.compile(r"^\s*\d{1,2}-\d{1,3}\s*$")
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
    tables: list[Table] = field(default_factory=list)   # 工单03：结构化表格
    figures: list[dict] = field(default_factory=list)    # 工单04：图区域 + 语义描述
    doc_name: str = ""                                   # 人类可读文档名（用于区分答的是哪家）

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages if p.text)


def _clean_line(line: str) -> str:
    """去掉行内多余空白，保留中文原样。"""
    line = line.replace("\u3000", " ")  # 全角空格
    line = re.sub(r"[ \t]{2,}", " ", line)
    return line.strip()


def _is_noise(line: str, header_res: list[re.Pattern] | None = None) -> bool:
    if not line:
        return True
    for rx in (header_res or _HEADER_RES_DEFAULT):
        if rx.match(line):
            return True
    if _PAGENO_RE.match(line) or _PAGENO_RE2.match(line) or _BARE_PAGENO_RE.match(line):
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


def _drop_key(s: str) -> str:
    """剔除判定用的行键：**只去空白，不做其他归一化**。

    一定要去空白：pypdf 提的是「2008 年中国 IC 市场应用结构与增长(亿元)」，
    pymupdf 提的是「2008 年中国IC 市场应用结构与增长(亿元)」，空格位置不同；
    但也不能做全角转半角之类的强归一化，否则「（3）」和「(3)」会被误判成同一行，
    把正文里的编号标题一起删掉。
    """
    return re.sub(r"\s+", "", s)


def clean_page_text(raw: str, split_inline: bool = True,
                    header_res: list[re.Pattern] | None = None,
                    drop_lines: set[str] | None = None) -> str:
    """
    清洗单页文本：逐行去噪 + 行内标题断行 + 可选剔除指定行。

    split_inline=False 时**跳过行内标题断行**——这是消融实验的开关：
    朴素基线直接取页文本，不做这一步。

    header_res：按文档传入的页眉正则（多文档语料必需，写死一家公司会漏另一家）。

    drop_lines（工单04）：要从正文里剔除的行（归一化空白后的形式）。
    来源于图区域检测 —— 包含两类：
      ① **图内文字**：p39 组织图的竖排节点名被 PDF 文字层逐字打散成
         「财」「务部」「销」「售」「处」，留在正文里会形成**看着像有答案、
         实际归属关系全错**的误导块（比如把"市场开发部/信息系统部/行政人事部"
         挤成一行），生成模型读到会照着这段答，比没有更糟；
      ② **图题**：工单3「坑 13」在图像上的复现。p310 的图题
         「2008年中国IC市场应用结构与增长(亿元)」与 id=6 的问题字面重合度极高，
         留着它一定会排第一，把真正的图像块挤出候选。
    """
    drops = drop_lines or set()
    lines = [_clean_line(ln) for ln in raw.splitlines()]
    out: list[str] = []
    for ln in lines:
        if not ln:
            continue
        if _is_noise(ln, header_res):
            continue
        if drops and _drop_key(ln) in drops:
            continue
        out.append(ln)
    joined = "\n".join(out).strip()
    return split_inline_headings(joined) if split_inline else joined


# ------------------------------------------------------------------ 表格抽取
#
# 工单03 起，表格解析全部交给 src/table_parser.py：
#   * 不再用「一行里出现 ≥3 个数字」猜表格 —— 正是那个启发式漏掉了第 157 页
#     的关联方表（全表只有一个 ``42.35%``），而工单验收题 id=3/id=4 的答案就在那张表里；
#   * 不再直接用 extract_tables() 转 Markdown —— 双线边框会被当成列，
#     三列的表切出九列，表头与数据错位。
# 实测全量跑 find_tables 的代价可以接受：力源 350 页 15.3s、兴图 548 页 51.4s。


def parse_pdf(
    pdf_path: Path,
    with_tables: bool = True,
    split_inline: bool = True,
    header_res: list[re.Pattern] | None = None,
    doc_name: str = "",
    table_gap_pt: float | None = None,
    table_max_rows: int | None = None,
    figure_drops: dict[int, list[str]] | None = None,
    figures: list[dict] | None = None,
) -> ParsedDoc:
    """
    解析 PDF。返回按页组织的文本；with_tables=True 时另外抽结构化表格。

    with_tables=False 是**表格解析消融的对照组**：表格内容不做结构化，
    而是作为普通文本行参与分块（也就是工单1/2 的「把 PDF 切碎了直接塞向量库」做法）。

    figure_drops / figures（工单04）：
      figure_drops = {页码: [要从正文剔除的行]}，由 scripts/build_images.py 产出；
      figures      = 图清单（含多模态语义描述），会挂到 ParsedDoc.figures 上供分块器成块。
      传 figures=None 表示**不做图像解析**（工单4 消融的对照组）。

    单页解析失败不中断整份文档（工单「容错机制」要求）。
    """
    from pypdf import PdfReader

    pdf_path = Path(pdf_path)
    reader = PdfReader(str(pdf_path))
    pages: list[PageText] = []

    drops_by_page: dict[int, set[str]] = {}
    for pg, lines in (figure_drops or {}).items():
        drops_by_page[int(pg)] = {_drop_key(x) for x in lines}

    for i, page in enumerate(reader.pages):
        try:
            raw = page.extract_text() or ""
        except Exception:  # noqa: BLE001  单页失败不应中断整份文档
            raw = ""
        text = clean_page_text(raw, split_inline=split_inline, header_res=header_res,
                               drop_lines=drops_by_page.get(i + 1))
        pages.append(PageText(page=i + 1, text=text))

    tables: list[Table] = []
    if with_tables:
        page_texts = {p.page: p.text for p in pages}
        tables, outside = parse_pdf_tables(
            pdf_path, doc=doc_name or pdf_path.stem,
            page_texts=page_texts, want_text_outside=True,
        )
        # 把「表格区域」从正文里摘掉：表格内容只以结构化表格的形态进索引，
        # 避免同一批事实以「结构化块 + 稀释长段落」两种质量同时存在。
        # 对照版（with_tables=False）不做这一步 —— 表格内容全部当普通文本行。
        for p in pages:
            if p.page in outside:
                # 走一遍同一套清洗（去页眉页码 + 行内标题断行 + 删图内文字），
                # 保证「有表格的页」和「没表格的页」的正文处理口径一致
                p.text = clean_page_text(outside[p.page], split_inline=split_inline,
                                         header_res=header_res,
                                         drop_lines=drops_by_page.get(p.page))

    return ParsedDoc(source=pdf_path.name, total_pages=len(pages), pages=pages,
                     tables=tables, figures=list(figures or []),
                     doc_name=doc_name or pdf_path.stem)


def parse_pdf_text_only(pdf_path: Path) -> ParsedDoc:
    """只做文字解析（不抽表格），用于快速干跑与分块回归。"""
    return parse_pdf(pdf_path, with_tables=False)
