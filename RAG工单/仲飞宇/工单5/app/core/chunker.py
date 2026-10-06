# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
# 工单编号：人工智能NLP-RAG-图像内容解析及检索优化
# 工单01 - 基于PDF文档的问答系统
# 工单04 - 图像内容解析及检索优化
"""
分块器：句子边界优先 + 表格/图像原子化。

【为什么不是固定 500 字切块】
实测 p128（印刷页码 1-1-128）有这样一个句子，137 字，同时装着两个
不同提问的答案：

    「报告期内，公司来自军用领域的收入分别为6,464.51万元、14,414.16万元、
      18,780.67万元和4,627.14万元，占主营业务收入比重分别为82.10%、
      97.31%、94.84%和94.34%。」

「军用领域各期收入是多少」和「军品收入占比是多少」都指向它。
按固定字数切，切点落在句子中间是**必然**的（137 字跨在 500 字边界上的
概率很高），一旦切断，两个问题会同时检索不到完整答案。
因此：**先按句切，再按目标长度聚合，绝不切开一个句子。**

【表头】招股书大量「键值对」表，整表 + 表题是不可切分的原子 chunk，
理由见 pdf_parser.TableBlock.to_markdown() 的注释。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Iterator

from app.config import settings
from app.core.pdf_parser import PageContent, TableBlock, TextBlock

# ----------------------------------------------------------------------
# 章节标题识别：用于给 chunk 打上 section_path（工单02 small-to-big 的地基）
# ----------------------------------------------------------------------
_HEAD_PATTERNS: list[tuple[re.Pattern[str], int]] = [
    (re.compile(r"^第[一二三四五六七八九十百]+节"), 0),
    (re.compile(r"^[一二三四五六七八九十百]+、"), 1),
    (re.compile(r"^（[一二三四五六七八九十百]+）"), 2),
    (re.compile(r"^\d+[．.、]\s*\S"), 3),
]

# 句末标点。分号也算边界 —— 招股书里分号常用于并列长句，切开更利于检索，
# 且不会破坏「5,520.00」这类数字（小数点不是中文句号）。
_SENT_END = re.compile(r"(?<=[。！？；])")
# 列表项前缀，形如「（1）」「①」「1、」——它们在句中充当新句的起点
_LIST_MARK = re.compile(r"(?<=[。；\n])\s*(?=(?:（\d+）|[①-⑳]|\d+[．.、]))")

# 目录页特征：点线引导符 + 全程无句末标点。
# 实测「六、发行人选择的具体上市标准 .......... 26」这类行会让 chunk 长到 2120 字
# 且没有任何完整句子，检索时纯属噪声（会挤占 top-k 名额）。
_TOC_LEADER = re.compile(r"\.{4,}|…{2,}|·{4,}")


def _is_toc(text: str) -> bool:
    """判断是否为目录（点线引导）文本。"""
    if "。" in text or "；" in text:
        return False
    return len(_TOC_LEADER.findall(text)) >= 2


@dataclass
class Chunk:
    """入库的最小单元。"""

    content: str
    chunk_type: str            # text | table | image（工单04）
    page_no: int               # 起始页的 PDF 索引
    page_label: str            # 起始页的印刷页码
    chunk_index: int = 0       # 文档内序号
    section_path: str = ""
    pages: list[int] = field(default_factory=list)   # 覆盖到的所有页索引
    meta: dict = field(default_factory=dict)

    @property
    def n_chars(self) -> int:
        return len(self.content)


def _is_heading(text: str) -> tuple[int, str] | None:
    t = text.strip()
    if len(t) > 60:        # 标题不会太长，避免把正文误判为标题
        return None
    for pat, level in _HEAD_PATTERNS:
        if pat.match(t):
            return level, t
    return None


def _split_sentences(text: str) -> list[str]:
    """按句末标点切句，保留标点；再按列表项前缀补切。"""
    parts: list[str] = []
    for seg in _SENT_END.split(text):
        seg = seg.strip()
        if not seg:
            continue
        parts.extend(p.strip() for p in _LIST_MARK.split(seg) if p.strip())
    return parts


class SectionTracker:
    """维护「第X节 > 一、 > （一）」层级路径。"""

    def __init__(self) -> None:
        self.stack: list[str] = []

    def feed(self, text: str) -> None:
        hit = _is_heading(text)
        if not hit:
            return
        level, title = hit
        del self.stack[level:]
        while len(self.stack) < level:
            self.stack.append("")
        self.stack.append(title)

    @property
    def path(self) -> str:
        return " > ".join(s for s in self.stack if s)


# ----------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------
def _blocks_in_reading_order(page: PageContent) -> list[TextBlock]:
    """按阅读顺序（先上后下、先左后右）排列文本块，并剔除纯页码类残留。"""
    blocks = [b for b in page.texts if b.text.strip()]
    blocks.sort(key=lambda b: (round(b.bbox[1] / 4), b.bbox[0]))
    return blocks


def _segments_of_page(page: PageContent) -> list[tuple[str, int, str]]:
    """
    把一页的文本块整理成「段」序列。

    【为什么必须合并连续的正文块】
    实测 p128 的金句在 PDF 里被拆成 4 个文本块（PDF 的块边界与句子边界无关）：

        块1: 「…公司来自军用领域的」
        块2: 「收入分别为6,464.51 万元、14,414.16 万元、18,780.67 万元和4,627.14 万元，占」
        块3: 「主营业务收入比重分别为82.10%、97.31%、94.84%和94.34%。」

    如果对每个块单独切句，块 1、块 2 里根本没有句末标点，会被当成两个
    「句子」，句子的完整性只能靠运气维持。
    所以：**先把连续的正文块拼成一段**（标题除外，标题本身是自然边界），
    再对整段切句。这样跨块的句子会被自动接回去。

    返回 [(文本, 页索引, 页标签), …]，标题与正文段都已区分好。
    """
    segs: list[tuple[str, int, str]] = []
    run: list[str] = []

    def push_run() -> None:
        if run:
            segs.append(("".join(run), page.page_no, page.page_label))
            run.clear()

    for b in _blocks_in_reading_order(page):
        if _is_heading(b.text):
            push_run()                      # 标题是边界，先落掉前面的正文段
            segs.append((b.text, b.page_no, b.page_label))
        else:
            run.append(b.text)
    push_run()

    # 丢弃目录段（点线引导、无完整句子），只保留有内容的段
    return [s for s in segs if not _is_toc(s[0])]


def chunk_pages(
    pages: Iterable[PageContent],
    *,
    min_chars: int | None = None,
    max_chars: int | None = None,
    overlap_sentences: int | None = None,
) -> Iterator[Chunk]:
    """
    把页流切成 chunk。

    文本 chunk 允许跨页（章节连续时更自然），但会记录覆盖的所有页码，
    以便前端引用时显示「1-1-128 ~ 1-1-129」。
    """
    min_chars = min_chars or settings.chunk_min_chars
    max_chars = max_chars or settings.chunk_max_chars
    overlap_sentences = (settings.chunk_overlap_sentences
                         if overlap_sentences is None else overlap_sentences)

    tracker = SectionTracker()
    idx = 0

    # 待聚合的句子缓冲：(句子, 页索引, 页标签)
    buf: list[tuple[str, int, str]] = []
    buf_chars = 0

    def flush(force: bool = False) -> Iterator[Chunk]:
        nonlocal buf, buf_chars, idx
        if not buf:
            return
        if not force and buf_chars < min_chars:
            return
        content = "".join(s for s, _, _ in buf)
        pages_hit = sorted({p for _, p, _ in buf})
        idx += 1
        yield Chunk(
            content=content,
            chunk_type="text",
            page_no=buf[0][1],
            page_label=buf[0][2],
            chunk_index=idx,
            section_path=tracker.path,
            pages=pages_hit,
        )
        # 重叠：保留末尾若干句，避免跨 chunk 的答案被边界切断
        keep = buf[-overlap_sentences:] if overlap_sentences else []
        buf = list(keep)
        buf_chars = sum(len(s) for s, _, _ in buf)

    for page in pages:
        for text, pno, plabel in _segments_of_page(page):

            # ---- 章节标题：影响后续 chunk 的 section_path，本身也进库 ----
            if _is_heading(text):
                tracker.feed(text)          # 在打 section_path 之前更新层级
                buf.append((text, pno, plabel))
                buf_chars += len(text)
                continue

            for sent in _split_sentences(text):
                # 兜底：个别段落整段无句末标点（如表格化排版的段落），
                # 单句就可能超过上限。此时只能硬切 —— 属于病态输入，
                # 正常语料下不会触发（实测 1139 个 chunk 中 0 例）。
                if len(sent) > max_chars:
                    if buf and buf_chars >= min_chars:
                        yield from flush()
                    for i in range(0, len(sent), max_chars):
                        piece = sent[i:i + max_chars]
                        buf.append((piece, pno, plabel))
                        buf_chars += len(piece)
                        if buf_chars >= max_chars:
                            yield from flush()
                    continue

                # 若加入这句会超上限，且当前缓冲已达下限 → 先落盘
                if buf and buf_chars + len(sent) > max_chars and buf_chars >= min_chars:
                    yield from flush()
                buf.append((sent, pno, plabel))
                buf_chars += len(sent)

        # 页末：够长就落盘，否则跨页继续累积
        if buf_chars >= min_chars:
            yield from flush()

        # ---- 表格：原子 chunk，立即输出 ----
        for tb in page.tables:
            idx += 1
            yield Chunk(
                content=tb.to_markdown(),
                chunk_type="table",
                page_no=tb.page_no,
                page_label=tb.page_label,
                chunk_index=idx,
                section_path=tracker.path,
                pages=[tb.page_no],
                meta={"title": tb.title, "n_rows": tb.n_rows, "n_cols": tb.n_cols},
            )

        # ---- 图（工单04）：同样原子 chunk，紧跟表格输出 ----
        # 【为什么也放在页内、表格之后】chunk_index 是全文档递增计数器，
        # 图块必须落在**它自己那一页的位置**上，否则 `chunk_index±1` 的邻块
        # 扩展会把别的页的块当作它的邻居，前端 `#chunk_index` 的排序也会失真。
        for fig in page.figures:
            if not fig.text and not fig.labels and not fig.title:
                continue          # 既没转写、也没图内文字、连题注都没有：不留空块
            idx += 1
            yield Chunk(
                content=compose_figure_content(fig, tracker.path),
                chunk_type="image",
                page_no=fig.page_no,
                page_label=fig.page_label,
                chunk_index=idx,
                section_path=tracker.path,
                pages=[fig.page_no],
                meta={"title": fig.title, "kind": fig.kind,
                      "figure_bbox": list(fig.bbox),
                      "vlm": bool(fig.text), "reviewed": bool(fig.reviewed),
                      "n_src_boxes": len(fig.src_bboxes)},
            )

    yield from flush(force=True)


def compose_figure_content(fig, section_path: str = "") -> str:
    """图块正文。**纯函数**，可以脱离 PDF 单测。

    形态（刻意让题注顶格加粗，与 TableBlock.to_markdown 的写法一致）：

        **2008年中国IC市场应用结构与增长(亿元)**
        【图像内容】…
        【图内文字】…

    【为什么题注缺失时要退回 section_path】实测书2 页 38 的组织结构图，
    它的真题注「1、公司组织结构图」印在**上一页**（页 37），而页内 40pt 内
    最近的文本块是图里的方框名。这时至少要给块一个语义锚点，否则它是一堆
    没有出处的方框名，检索引擎无从判断它属于哪一节。section_path 来自
    章节追踪器（页 38 上是「… > 四、发行人组织结构 > 1、公司组织结构图」）。
    """
    lines: list[str] = []
    anchor = fig.title or (section_path.split(" > ")[-1] if section_path else "")
    if anchor:
        lines.append(f"**{anchor}**")
        lines.append("")
    if fig.text:
        lines.append(f"【图像内容】{fig.text}")
    # 图内文本层文字（仅结构图有；页 38 实测 17 个标签与原文逐字一致）
    if fig.labels and fig.labels not in (fig.text or ""):
        lines.append(f"【图内文字】{fig.labels}")
    return "\n".join(lines).strip()


#: 原子块类型：整块不可切窗口、邻块无条件收（见 retriever）
ATOMIC_CHUNK_TYPES = ("table", "image")


def is_atomic_chunk(chunk_type: str) -> bool:
    """该类型的 chunk 是不是"整块不可切分"的原子块。

    【为什么 table 与 image 同类】两者都是**语义密度高、条目式**的内容：
    按窗口截断会切在条目中间，静默丢掉答案（工单01 的 id=793 就是这么挂的）；
    而且它们与"引子句"常常没有共同查询词，会被互补性过滤误伤。
    """
    return chunk_type in ATOMIC_CHUNK_TYPES


def table_to_chunk(tb: TableBlock, idx: int, section_path: str = "") -> Chunk:
    """单个表格 → chunk（供单测使用）。"""
    return Chunk(
        content=tb.to_markdown(),
        chunk_type="table",
        page_no=tb.page_no,
        page_label=tb.page_label,
        chunk_index=idx,
        section_path=section_path,
        pages=[tb.page_no],
    )
