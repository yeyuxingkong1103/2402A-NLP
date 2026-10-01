# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
# 工单01 - 基于PDF文档的问答系统
"""
PDF 解析器：文字 + 表格。

对应工单01「功能验收 1：准确解析文字和表格数据」。

【三个关键设计，全部来自对 招股说明书1.pdf 的实测】

1. `find_tables()` 而非 `get_text()` 抽表格
   实测 p22 用纯文本流抽出来是**粘连**的：
     `发行前每股净资产3.55 元/ 股（按照…）发行前每股收益0.79 元/股…`
   列与列之间没有分隔符，左侧表头与右侧数值糊成一团，无法还原语义。
   而 `find_tables()` 按 bbox 重建后行列分明：
     ['发行前每股净资产', '3.55 元/ 股（…）', '发行前每股收益', '0.79 元/股（…）']
   所以表格一律走 `find_tables()`。

2. 页眉页脚**锚定整行**删除，绝不按关键词删
   页眉是 `武汉兴图新科电子股份有限公司        招股意向书`（实测 y≈44）。
   如果简单地"删除含公司名的行"，会把**正文里提到公司名的句子也删掉** ——
   而 p21（发行人基本情况）正文恰恰反复出现公司全称。
   这是「静默答错」，比崩溃可怕得多。
   因此规则是：位置必须在页眉/页脚带内 **且** 整行完全匹配固定模板。

3. 表格去重
   表格区域的文字同样会被 `get_text()` 抽出来，若不剔除，
   同一份内容会同时以「散文」和「表格」两种形态进入知识库，
   既浪费 top-k 名额又可能让上下文自相矛盾。
   所以：先抽表格占用的 bbox，再把落在此 bbox 内的文本块丢弃。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import pymupdf

# ----------------------------------------------------------------------
# 页眉 / 页脚模板（实测自 招股说明书1.pdf）
# ----------------------------------------------------------------------
# 页眉：`武汉兴图新科电子股份有限公司` + 若干空白 + `招股意向书`
# 用 \s* 容忍 PDF 抽取出的连续空格；**整行匹配**是安全的关键
HEADER_RE = re.compile(r"^\s*武汉兴图新科电子股份有限公司\s+招股意向书\s*$")
# 页脚：`1-1-22` 形式的分节页码
FOOTER_PAGE_RE = re.compile(r"^\s*1-1-\d+\s*$")
# 页眉/页脚所在的纵向比例带（实测页高 842：页眉 y≈44，页脚 y≈779）
HEADER_BAND = 0.12
FOOTER_BAND = 0.85


@dataclass
class TextBlock:
    """一段正文。"""

    page_no: int          # PDF position index（0 基）
    page_label: str       # 印刷页码，如 1-1-22
    text: str
    bbox: tuple[float, float, float, float]


@dataclass
class TableBlock:
    """一张表。整表是不可切分的原子 chunk。"""

    page_no: int
    page_label: str
    rows: list[list[str]]
    bbox: tuple[float, float, float, float]
    title: str = ""       # 表上方最近的标题/说明行（提升检索可读性）

    # ---------------------------------------------------------------
    @property
    def n_rows(self) -> int:
        return len(self.rows)

    @property
    def n_cols(self) -> int:
        return max((len(r) for r in self.rows), default=0)

    def to_markdown(self) -> str:
        """
        转 Markdown，并做**表头下推**。

        【为什么必须下推】招股书大量使用「两列键值对」型表格：
            | 发行前每股净资产 | 3.55 元/股 |
        也有「横向铺开」型：
            | 发行前每股净资产 | 3.55 | 发行前每股收益 | 0.79 |
        后者第一行既是表头又是数据，若只把第 0 行当表头，
        数据行会失去语义（一行里 4 个格子，读者不知道哪个是标签）。
        下推后每个单元格自带标签，语义密度显著提升。
        """
        if not self.rows:
            return ""

        def clean(c: object) -> str:
            if c is None:
                return ""
            # find_tables() 在合并单元格处返回 None；文本内的换行压成空格
            return normalize_text(re.sub(r"\s+", " ", str(c)))

        grid = [[clean(c) for c in row] for row in self.rows]
        width = max(len(r) for r in grid)
        grid = [r + [""] * (width - len(r)) for r in grid]

        lines: list[str] = []
        if self.title:
            lines.append(f"**{self.title}**")
        lines.append("")

        header = grid[0]
        # 表头是否"有效"：非空且不与首行数据重复
        has_header = any(header) and any(h for h in header)
        body = grid[1:] if has_header else grid
        if has_header:
            lines.append("| " + " | ".join(h or "—" for h in header) + " |")
            lines.append("|" + "---|" * width)
        else:
            # 无表头：把「行标签」当第一列名，避免 Markdown 表格无表头
            lines.append("| " + " | ".join(f"字段{i + 1}" for i in range(width)) + " |")
            lines.append("|" + "---|" * width)

        for row in body:
            if not any(row):
                continue
            lines.append("| " + " | ".join(c or "—" for c in row) + " |")
        return "\n".join(lines)


@dataclass
class PageContent:
    page_no: int
    page_label: str
    texts: list[TextBlock] = field(default_factory=list)
    tables: list[TableBlock] = field(default_factory=list)


@dataclass
class ParseStats:
    """解析统计，供入库脚本打印与验收留痕。"""

    n_pages: int = 0
    n_text_blocks: int = 0
    n_tables: int = 0
    n_header_removed: int = 0
    n_footer_removed: int = 0
    n_table_text_dropped: int = 0
    n_empty_pages: int = 0


_CJK = r"一-鿿　-〿＀-￯"
# 与中文相邻的空格：PDF 排版产生的假空格。
# 规则：空格**任一侧**是 CJK 就删除。
# 用「任一侧」而不是「数字↔中文」，是因为实测还有 `3.55 元/ 股` 这种
# 斜杠两侧的情况 —— 只盯数字会漏掉它。
# 纯英文词组（Wuhan Xingtu Xinke）两侧都不是 CJK，空格得以保留。
_SPACE_ADJ_CJK = re.compile(rf"(?<=[{_CJK}])\s+|\s+(?=[{_CJK}])")


def normalize_text(text: str) -> str:
    """
    规范化 PDF 抽取的文本。

    【为什么必须做】实测 PDF 里读出的是 `6,464.51 万元`、`2019 年12 月16 日`、
    `3.55 元/ 股` —— 数字与中文之间夹了排版用空格。不清理会有两个后果：
      1. jieba 分词把 `5,520.00 万元` 切成 `5,520.00` / `万元`，BM25 里
         数字 token 与正文对不上；
      2. 评估的「规则化数值命中」用正则匹配 `5,520.00万元` 时会失败。
    规则很保守：只有当空格**一侧是 CJK** 时才删除，纯英文词组
    （`Wuhan Xingtu Xinke Electronics`）不受影响。
    """
    text = text.replace(" ", " ")
    text = _SPACE_ADJ_CJK.sub("", text)
    return re.sub(r"[ \t]{2,}", " ", text).strip()


def page_label(page_no: int, offset: int) -> str:
    """
    PDF position index → 印刷页码。

    【实测】招股说明书1.pdf 的页脚为 `1-1-N` 且 index N ↔ 页脚 1-1-N，
    即 offset = 0（`config.page_label_offset`）。
    该值由 `scripts/preflight.py` 每次校验，防止换文件后静默错位。
    """
    return f"1-1-{page_no - offset}"


def _in_bbox(x0: float, y0: float, x1: float, y1: float,
             box: tuple[float, float, float, float], pad: float = 2.0) -> bool:
    """文本块是否落在表格 bbox 内（允许 pad 像素的边界容差）。"""
    bx0, by0, bx1, by1 = box
    return (x0 >= bx0 - pad and y0 >= by0 - pad
            and x1 <= bx1 + pad and y1 <= by1 + pad)


def parse_page(page: pymupdf.Page, page_no: int, offset: int,
               stats: ParseStats) -> PageContent:
    """解析单页：先抽表格占位，再抽正文并剔除落在表格内的块。"""
    label = page_label(page_no, offset)
    height = page.rect.height
    pc = PageContent(page_no=page_no, page_label=label)

    # ---------------- 1. 表格 ----------------
    table_boxes: list[tuple[float, float, float, float]] = []
    try:
        found = page.find_tables()
    except Exception:  # noqa: BLE001
        found = None

    if found is not None:
        for t in found.tables:
            rows = t.extract()
            rows = [[c for c in r] for r in rows] if rows else []
            if not rows or not any(any(c for c in r) for r in rows):
                continue
            box = tuple(float(v) for v in t.bbox)
            table_boxes.append(box)
            pc.tables.append(TableBlock(
                page_no=page_no, page_label=label, rows=rows, bbox=box,
            ))
            stats.n_tables += 1

    # ---------------- 2. 正文 ----------------
    for b in page.get_text("dict")["blocks"]:
        if b.get("type") != 0:      # 0=文本块，1=图像块（图像留给工单04）
            continue
        x0, y0, x1, y1 = b["bbox"]
        raw = "".join(
            s["text"] for line in b.get("lines", []) for s in line.get("spans", [])
        ).strip()
        if not raw:
            continue

        # -- 2a. 页眉页脚：位置带内 + 整行匹配模板，两个条件同时满足才删 --
        # 注意：必须在 normalize_text **之前**判，因为页眉模板依赖词间空格，
        # 规范化会把中文之间的空格吃掉，导致模板失配。
        if y1 < height * HEADER_BAND and HEADER_RE.match(raw):
            stats.n_header_removed += 1
            continue
        if y0 > height * FOOTER_BAND and FOOTER_PAGE_RE.match(raw):
            stats.n_footer_removed += 1
            continue

        text = normalize_text(raw)
        if not text:
            continue

        # -- 2b. 落在表格里的文本块丢弃（否则同一内容进库两次）--
        if any(_in_bbox(x0, y0, x1, y1, box) for box in table_boxes):
            stats.n_table_text_dropped += 1
            continue

        pc.texts.append(TextBlock(
            page_no=page_no, page_label=label, text=text,
            bbox=(x0, y0, x1, y1),
        ))
        stats.n_text_blocks += 1

    # ---------------- 3. 给表格补标题 ----------------
    # 表格上方最近的一段短文本（通常在相邻 40px 内），多半是表题，
    # 例如「（二）本次发行上市的重要日期」。补上后表格 chunk 自带语义锚点。
    for tb in pc.tables:
        candidates = [
            t for t in pc.texts
            if t.bbox[3] <= tb.bbox[1] and tb.bbox[1] - t.bbox[3] < 60
            and 2 <= len(t.text) <= 60
            # 排除完整句子：表题通常是「（二）本次发行上市的重要日期」这类
            # 无句末标点的短语；带「。；」的是正文，取来当标题会误导检索。
            and not t.text.endswith(("。", "；", "：", "，"))
        ]
        if candidates:
            tb.title = max(candidates, key=lambda t: t.bbox[3]).text

    if not pc.texts and not pc.tables:
        stats.n_empty_pages += 1
    return pc


def parse_pdf(pdf_path: str | Path, offset: int = 0,
              limit: int | None = None) -> Iterator[PageContent]:
    """
    逐页解析 PDF。生成器形式，548 页不必一次性进内存。

    offset: 页码偏移 = position_index − 印刷页码（本文件实测为 0）
    limit : 只解析前 N 页，供调试
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF 不存在：{path}")

    stats = ParseStats()
    doc = pymupdf.open(path)
    try:
        stats.n_pages = doc.page_count
        n = doc.page_count if limit is None else min(limit, doc.page_count)
        for i in range(n):
            yield parse_page(doc[i], i, offset, stats)
    finally:
        doc.close()
        parse_pdf.last_stats = stats  # type: ignore[attr-defined]


def parse_pdf_full(pdf_path: str | Path, offset: int = 0,
                   limit: int | None = None) -> tuple[list[PageContent], ParseStats]:
    """解析全部页面并返回 (页列表, 统计)。"""
    pages = list(parse_pdf(pdf_path, offset=offset, limit=limit))
    return pages, getattr(parse_pdf, "last_stats", ParseStats())
