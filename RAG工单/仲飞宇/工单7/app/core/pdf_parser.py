# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
# 工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
# 工单编号：人工智能NLP-RAG-图像内容解析及检索优化
# 工单编号：人工智能NLP-RAG-功能测试及评估
# 工单04 - 图像内容解析及检索优化
# 工单07 - 功能测试及评估（页码改为「页脚实读优先」+ 按文档取偏移）
# 工单01 - 基于PDF文档的问答系统
# 工单03 - PDF文档的表格解析及检索优化
"""
PDF 解析器：文字 + 表格。

对应工单01「功能验收 1：准确解析文字和表格数据」。

【工单03 改了什么】页眉/页脚模板、页码格式原先写死成招股说明书1，现在一律从
`app/core/doc_profiles.py` 的 DocProfile 取（按文件名自动匹配，配不到的当场报错）。
招股书2 的页脚是**裸数字**、另有 8 页旋转 90° 的横向表，套用书1 的规则会一条都清不掉。

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

from app.config import settings
from app.core.doc_profiles import (
    DEFAULT_DOC, FOOTER_BAND, HEADER_BAND, DocProfile, profile_for_path,
)
from app.core.figure_locator import (
    Figure, LocateStats, locate_figures, scan_xref_pages,
)

# ----------------------------------------------------------------------
# 页眉 / 页脚模板：**按文档配置**，见 app/core/doc_profiles.py
# ----------------------------------------------------------------------
# 原先这里写死的是招股说明书1.pdf 的模板（兴图新科 + `1-1-N` 页码）。
# 加工单03 的第二份文档时实测：那份文档页脚是**裸数字**、公司名也不同，
# 于是页眉页脚**一条都清不掉**（实测各清 0 条），而且没清掉的页眉还会被
# 下面的"表格补标题"启发式抓走，变成一张表的标题。
# 现在统一从 DocProfile 取；分带常量也在 doc_profiles 里。


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
            s = normalize_text(re.sub(r"\s+", " ", str(c)))
            # 【工单03 修】单元格里可能出现**字面竖线**（实测招股书1 第 1-1-20 页
            # 的术语表里有一处 `参数扰动|摄动`）。不转义的话它会被当成 Markdown 的
            # 列分隔符，那一行就比别的行多一列 —— 整张表结构错乱、渲染崩掉。
            # 转义成 `\|` 是标准做法，且不丢信息。
            return s.replace("|", r"\|")

        grid = [[clean(c) for c in row] for row in self.rows]
        width = max(len(r) for r in grid)
        grid = [r + [""] * (width - len(r)) for r in grid]

        # ---------- 工单03：先修「合并表头错位」，再砍空列 ----------
        # 【错位是实测现象】find_tables() 遇到合并表头时，会把表头的值放在一组列、
        # 数据行的值放在**另一组**列。实测第 21 页募投表：
        #     表头：['', '序号', '', '', '项目名称', '', '', '计划总投资(万元)', '']
        #     数据：['1', None, None, '仓储及物流中心', None, None, '3,393.40', None, None]
        # 表头在列 1/4/7、数据在列 0/3/6 —— 整体错开一列。
        # 不修的话，砍空列会把两组列**都**留下（因为各自都有值），
        # 渲染出来就是「序号」压在「—」上、数据和表头对不齐。
        #
        # 修法：当表头的非空列**不在**数据行的非空列里（即确实错位）、
        # 且两边个数相同时，按出现顺序把表头的值重新锚到数据的列上。
        # 正常对齐的表（表头列 ⊆ 数据列）条件不成立，一律不动。
        if len(grid) >= 2:
            hdr_cols = [c for c in range(width) if grid[0][c].strip()]
            dat_cols = [c for c in range(width)
                        if any(grid[r][c].strip() for r in range(1, len(grid)))]
            if hdr_cols and dat_cols and len(hdr_cols) == len(dat_cols) \
                    and not set(hdr_cols) <= set(dat_cols):
                realigned = [""] * width
                for hc, dc in zip(hdr_cols, dat_cols):
                    realigned[dc] = grid[0][hc]
                grid[0] = realigned

        # ---------- 工单03：砍掉**整列全空**的列 ----------
        # 【为什么必须砍】find_tables() 把合并单元格展开成若干"跨度列"，
        # 只有 1 列有值、其余全空。实测招股说明书2.pdf 的 9 张关键表里，
        # 346 个单元格有 204 个是空的（59%）。这些空列在 Markdown 里变成一串 `—`，
        # 把表格的语义密度稀释掉，直接拖累检索。
        #
        # 第 21 页募投表：           砍之前（9 列，59% 是 —）
        #   | — | 序号 | — | — | 项目名称 | — | — | 计划总投资(万元) | — |
        #   | 1 | — | — | 仓储及物流中心 | — | — | 3,393.40 | — | — |
        # 砍之后（3 列）：
        #   | 序号 | 项目名称 | 计划总投资(万元) |
        #   | 1 | 仓储及物流中心 | 3,393.40 |
        #
        # 【安全边界】只在**整列全空**时删 —— 该列只要在**任一行**有值就保留。
        # 这是"宁可留噪声也不丢数据"的一侧；真正有值的列一个都不会动。
        keep = [c for c in range(width) if any(r[c].strip() for r in grid)]
        if keep and len(keep) < width:
            grid = [[r[c] for c in keep] for r in grid]
            width = len(keep)
        if not width:
            return ""

        lines: list[str] = []
        if self.title:
            lines.append(f"**{self.title}**")
        lines.append("")

        header = grid[0]
        # 表头是否"有效"：非空则认为是表头
        has_header = any(header)
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
            # 【注意】这里对**单元格内**的空仍填 `—`：局部空值是位置信息，
            # 删掉会让同一行的值与列错位。只有"整列全空"才删列（上面那步）。
            lines.append("| " + " | ".join(c or "—" for c in row) + " |")
        return "\n".join(lines)


@dataclass
class PageContent:
    page_no: int
    page_label: str
    texts: list[TextBlock] = field(default_factory=list)
    tables: list[TableBlock] = field(default_factory=list)
    # 工单04：图表 / 组织结构图。几何与题注在这一步定下来，
    # 语义文本（figure.text）由 pipeline 的 figure 阶段用多模态模型回填。
    figures: list[Figure] = field(default_factory=list)


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
    # ---------- 工单04：图区 ----------
    n_figures: int = 0
    n_figures_bitmap: int = 0
    n_figures_vector: int = 0
    n_images_dropped: int = 0      # 被四道规则滤掉的位图实例（噪声）


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


def page_label(page_no: int, offset: int = 0,
               doc: DocProfile | None = None) -> str:
    """
    PDF position index → 印刷页码。

    【实测】两份文档的偏移都是 0（index N ↔ 页脚 N），但**格式不同**：
      招股说明书1.pdf → `1-1-N`（分节页码）    招股说明书2.pdf → `N`（裸数字）
    所以格式从 DocProfile 取；`doc=None` 时用默认文档（保持工单01/02 的既有行为）。
    偏移值由 `scripts/preflight.py` 每次校验，防止换文件后静默错位。
    """
    return (doc or DEFAULT_DOC).label(page_no, offset)


def _in_bbox(x0: float, y0: float, x1: float, y1: float,
             box: tuple[float, float, float, float], pad: float = 2.0) -> bool:
    """文本块是否落在表格 bbox 内（允许 pad 像素的边界容差）。"""
    bx0, by0, bx1, by1 = box
    return (x0 >= bx0 - pad and y0 >= by0 - pad
            and x1 <= bx1 + pad and y1 <= by1 + pad)


def parse_page(page: pymupdf.Page, page_no: int, offset: int,
               stats: ParseStats, doc: DocProfile | None = None,
               *, xref_page_count: dict[int, int] | None = None,
               prev_texts: list[TextBlock] | None = None,
               locate_stats=None) -> PageContent:
    """解析单页：先抽表格占位，再抽正文并剔除落在表格内的块，最后定位图区。

    xref_page_count / prev_texts 是**跨页状态**（工单04）：前者用于识别跨页
    重复的水印，后者用于"题注在上一页"的组织结构图。都由 parse_pdf 维护。
    """
    doc = doc or DEFAULT_DOC
    height = page.rect.height
    # 【工单07】页码改为「页脚实读优先」。实读要用 block 的 bbox，而下面的正文
    # 抽取也要同一份 blocks —— 只调一次 get_text("dict")，否则解析耗时翻倍
    # （实测 2812 页会平白多花约 6 分钟）。
    blocks = page.get_text("dict")["blocks"]
    label = doc.printed_label(page, blocks, doc.label(page_no, offset))
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
    for b in blocks:
        if b.get("type") != 0:      # 0=文本块，1=图像块（图像留给工单04）
            continue
        x0, y0, x1, y1 = b["bbox"]
        raw = "".join(
            s["text"] for line in b.get("lines", []) for s in line.get("spans", [])
        ).strip()
        if not raw:
            continue

        # -- 2a. 页眉页脚 --
        # 注意：必须在 normalize_text **之前**判，因为页眉模板依赖词间空格，
        # 规范化会把中文之间的空格吃掉，导致模板失配。
        stripped = raw.strip()

        # 页眉：**只看整行模板，不看分带**。
        # 模板是「公司全称 + 空白 + 招股意向书」占满整行，正文里不可能出现这种行，
        # 所以它天生安全。之所以去掉分带限制：实测招股说明书2.pdf 有 8 页是旋转 90°
        # 的横向表（842×595），页眉落在 y≈0.35，按分带判会被漏掉。
        if doc.matches_header(stripped):
            stats.n_header_removed += 1
            continue

        # 页脚：分带 + 整行模板，两个条件同时满足才删。
        # 【页脚**必须**保留分带】招股说明书2.pdf 的页脚模板是 `^\d{1,3}$`，
        # 也就是"任意裸数字"——脱离分带单独用会把正文里的数字全删掉
        # （实测第 136 页正文带内有 531 / 628 / 829 等 26 个裸数字块）。
        if y0 > height * FOOTER_BAND and doc.matches_footer(stripped):
            stats.n_footer_removed += 1
            continue

        # 页脚补漏：**整行恰好等于本页自己的页码**，不看分带。
        # 为的是上面那 8 页旋转横向页（页脚落在 y≈0.69，进不了页脚带）。
        #
        # 【工单07 收紧：只对非标准版面启用】这条规则**没有分带保护**，原注释里
        # 「不可能误伤别的数字」只对招股书成立。加工单07 的 9 份年报后实测反例：
        # 国泰君安 2019 年报 idx=10 的资质表里，有一格正文恰好就是「11」，
        # 而该页页码也是 11 → 整格被静默删掉。这正是本项目最忌讳的
        # 「不报错的错」。所以按它**当初存在的唯一理由**（旋转页/横版页的页脚
        # 落不进常规分带）加上版面条件 —— 招股书2 那 8 页旋转页照旧能清掉
        # （tests/test_pipeline.py::test_pdf2_header_footer_fully_removed 仍在守），
        # 而全部正排页面上这条规则不再生效，正文再也不会被它误删。
        if (bool(page.rotation) or page.rect.width > page.rect.height) \
                and stripped == label:
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

    # ---------------- 4. 图区（工单04）----------------
    # 【为什么放在最后】图区定位要用**已经清掉页眉页脚、已经剔除表格内文本**的
    # pc.texts 来判"竖排标签"和找题注 —— 拿原始页文本会把页码/表头也算进去。
    if settings.image_enable:
        pc.figures = locate_figures(
            page, pc.texts, pc.tables,
            page_no=page_no, page_label=label,
            xref_page_count=xref_page_count or {},
            prev_texts=prev_texts,
            stats=locate_stats,
        )
        stats.n_figures += len(pc.figures)
        stats.n_figures_bitmap = getattr(locate_stats, "n_bitmap", 0)
        stats.n_figures_vector = getattr(locate_stats, "n_vector", 0)
        stats.n_images_dropped = getattr(locate_stats, "n_images_dropped", 0)

    # 只有图的页不算空页（如组织结构图那一页的正文就是图）
    if not pc.texts and not pc.tables and not pc.figures:
        stats.n_empty_pages += 1
    return pc


def parse_pdf(pdf_path: str | Path, offset: int = 0,
              limit: int | None = None,
              doc_profile: DocProfile | None = None) -> Iterator[PageContent]:
    """
    逐页解析 PDF。生成器形式，548 页不必一次性进内存。

    offset      : 页码偏移 = position_index − 印刷页码（两份文档实测都为 0）
    limit       : 只解析前 N 页，供调试
    doc_profile : 文档配置。**不传则按文件名自动匹配**（见 doc_profiles），
                  匹配不到直接报错 —— 不静默套用别的文档的页眉页脚规则。
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF 不存在：{path}")

    prof = doc_profile or profile_for_path(path)
    stats = ParseStats()
    doc = pymupdf.open(path)
    try:
        stats.n_pages = doc.page_count
        n = doc.page_count if limit is None else min(limit, doc.page_count)
        # 【工单07】偏移改为「调用方给了就用调用方的，没给(0)就用该文档自己的」。
        # 工单01–06 的调用方一律传 settings.page_label_offset（=0），而两份招股书
        # 的 profile 偏移也正好是 0 —— 所以既有行为一字未变。
        # 工单07 的 9 份年报偏移各异（+1 / −1 / −3 / +20），只能按文档取。
        eff_offset = offset or prof.page_label_offset
        # 工单04：跨页状态。xref 预扫描只取图像清单（不抽文本），
        # 用来识别"每页都铺一遍"的水印；prev_texts 给"题注在上一页"的图兜底。
        xref_pages = scan_xref_pages(doc, n) if settings.image_enable else {}
        locate_stats = LocateStats()
        prev_texts: list[TextBlock] = []
        for i in range(n):
            pc = parse_page(doc[i], i, eff_offset, stats, prof,
                            xref_page_count=xref_pages,
                            prev_texts=prev_texts,
                            locate_stats=locate_stats)
            prev_texts = pc.texts
            yield pc
    finally:
        doc.close()
        parse_pdf.last_stats = stats  # type: ignore[attr-defined]


def parse_pdf_full(pdf_path: str | Path, offset: int = 0,
                   limit: int | None = None,
                   doc_profile: DocProfile | None = None
                   ) -> tuple[list[PageContent], ParseStats]:
    """解析全部页面并返回 (页列表, 统计)。"""
    pages = list(parse_pdf(pdf_path, offset=offset, limit=limit,
                           doc_profile=doc_profile))
    return pages, getattr(parse_pdf, "last_stats", ParseStats())
