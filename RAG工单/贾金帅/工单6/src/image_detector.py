"""
图像区域检测与裁剪模块（工单4 核心）
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
工单编号：人工智能NLP-RAG-图像内容解析及检索优化

为什么不能直接「把 PDF 里的图片抠出来」
----------------------------------------
工单备注要求「PDF 中的图像语义解析」，但招股说明书里的「图」并不是干净的一张图：

1) **一张图由几十个碎片图元拼成**。力源第 39 页的组织结构图由 29 个位图 + 37 条矢量
   连接线组成（每个节点框是一张独立的小位图），单独抠任何一个图元都只是一块灰底矩形。
2) **正文里的图片尺寸各异且夹杂装饰**。有的页只有一条页眉分隔线（矢量），
   有的是封面 logo，这些都不是「图」。
3) **图内文字是垃圾文本**。竖排节点名被 PDF 文字层逐字打散，pypdf 在 p39 上提出的是
   ``财`` / ``务部``、pymupdf 提出的是 ``财`` / ``务`` / ``部`` —— 与正文口径还不一致，
   拿它去和正文做字符串匹配剔重是做不干净的。

所以本模块的做法是**空间法**：
    a. 采集所有**位图 bbox**（`get_image_rects`）与**矢量绘图 bbox**（`get_drawings`），
       裁掉页眉/页码/页边距区域；
    b. 按间距做 **union-find 聚类**，把碎片图元合并成一个「图区域」；
       —— 节点框之间的 39~55pt 空隙由**连接线**（本身就是绘图矩形）搭桥，天然连通；
    c. 过滤掉尺寸过小（图标、装饰线）与面积过大的（整页底纹）区域；
    d. 图题单独识别：紧贴图区域上下 25pt 内、且水平有重叠的文本行；
    e. 产出**正文剔除掩码**：图内文字的字符多重集 + 图题行，
       供 pdf_parser 逐行判断"这行是不是图里的"。

第 (e) 步是工单3「坑 13」在图像上的复现：图题若留在正文，会形成一个
**有标题、无数据** 的碎片块。第 72 页的图题恰好是「2008年中国IC市场应用结构与增长(亿元)」，
与 id=6 的问题字面重合度极高，它一定会排第一、把真正的图像块挤出候选。
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

from .config import WORK_ORDER_NOS  # noqa: F401  (工单编号，模块标识)

# ------------------------------------------------------------------ 参数
GAP_PT = 30.0          # 图元聚类间距：两个图元扩张 GAP/2 后相交即视为同一张图
MIN_FIG_W = 90.0       # 最小图宽（pt）—— 滤掉 logo / 图标
MIN_FIG_H = 55.0       # 最小图高（pt）—— 滤掉页眉分隔线一类的细长装饰
MAX_FIG_AREA = 0.75    # 图区域面积占比上限（滤掉整页底纹）
MARGIN_TOP = 70.0      # 页眉区（公司名 + 分隔线）
MARGIN_BOTTOM = 62.0   # 页码区
MARGIN_X = 30.0
CAPTION_GAP_PT = 26.0  # 图题与图区域的最大间距
CROP_PAD_PT = 4.0      # 裁剪外扩，避免切掉边框线


# ------------------------------------------------------------------ 数据结构
@dataclass
class FigureRegion:
    """一个「图区域」：由若干位图/矢量图元聚类而成。"""

    page: int
    index: int                              # 页内序号，从 1 开始
    bbox: tuple[float, float, float, float]  # (x0, y0, x1, y1)，PDF 坐标（pt）
    kind: str = "raster"                    # raster / vector / mixed
    n_elems: int = 0                        # 合并了多少个图元（诊断用）
    caption: str = ""                       # 图题（含"资料来源"行）
    raster_path: str = ""                   # 裁剪出的 PNG（相对工程根目录）
    px_w: int = 0
    px_h: int = 0

    @property
    def area(self) -> float:
        return max(0.0, self.bbox[2] - self.bbox[0]) * max(0.0, self.bbox[3] - self.bbox[1])


@dataclass
class PageFigures:
    """一页的图区域集合 + 该页正文需要剔除的内容。"""

    page: int
    figures: list[FigureRegion] = field(default_factory=list)
    # 图内文字（把所有图区域内的字符收进一个多重集）
    inner_chars: dict[str, int] = field(default_factory=dict)
    # 图题行（归一化后，用于整行删除）
    caption_lines: set[str] = field(default_factory=set)


# ------------------------------------------------------------------ 工具
_WS_RE = re.compile(r"\s+")


def norm_line(s: str) -> str:
    """归一化：去所有空白 + 全角转半角。用于跨解析器的行比较。

    pypdf 提的是「2008年中国 IC 市场应用结构与增长(亿元)」，
    pymupdf 提的是「2008 年中国IC 市场应用结构与增长(亿元)」——
    空格位置不一致，必须先去空白再比。
    """
    s = unicodedata.normalize("NFKC", s)
    return _WS_RE.sub("", s)


def _rect_of(page, raw) -> tuple[float, float, float, float]:
    """兼容 pymupdf 的两种返回形态：Rect 对象（get_drawings / get_text("words")
    的 rect 字段）与 4 元组（get_text("dict") 的 bbox 字段）。"""
    if hasattr(raw, "x0"):
        return (float(raw.x0), float(raw.y0), float(raw.x1), float(raw.y1))
    return (float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3]))


def _intersect(a, b) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def _expand(r, pad: float):
    return (r[0] - pad, r[1] - pad, r[2] + pad, r[3] + pad)


def _overlap_area(a, b) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return max(0.0, w) * max(0.0, h)


def _clip_to_content(r, pw: float, ph: float):
    """裁到正文区域（去页眉/页码/页边距）。返回 None 表示不属于正文区。

    ⚠️ **必须容忍零宽/零高的退化矩形**。组织结构图的连接线就是这种形状：
    ``(153, 301, 423, 301)`` 是横线（高 0）、``(288, 99, 288, 138)`` 是竖线（宽 0）。
    早期版本用「裁剪后宽高都 > 1pt」做判据，把这些连接线全部丢掉了 ——
    结果是 p39 的 29 个节点框因为跨不过 39pt 的空隙而碎成 11 块、8 块被尺寸过滤掉，
    整张组织结构图只剩下底部三行残缺碎片。
    """
    x0, y0, x1, y1 = r
    cx0, cy0 = max(x0, MARGIN_X), max(y0, MARGIN_TOP)
    cx1, cy1 = min(x1, pw - MARGIN_X), min(y1, ph - MARGIN_BOTTOM)

    if cx1 - cx0 <= 1.0:            # 竖线（或被页边距截断）
        if MARGIN_X <= x0 and x1 <= pw - MARGIN_X:
            cx0, cx1 = x0, x1       # 整条线都在正文区内 → 保留原始坐标
        else:
            return None
    if cy1 - cy0 <= 1.0:            # 横线（或被页眉/页码截断）
        if MARGIN_TOP <= y0 and y1 <= ph - MARGIN_BOTTOM:
            cy0, cy1 = y0, y1
        else:
            return None

    if cx1 < cx0 or cy1 < cy0:
        return None
    return (cx0, cy0, cx1, cy1)


# ------------------------------------------------------------------ 聚类
def _inside_any(r, boxes, tol: float = 1.5, ratio: float = 0.6) -> bool:
    """矩形是否落在某个「表格区域」内。

    **这一步必不可少**：`get_drawings()` 会把**表格的线框**一并返回，
    如果不排除，带边框的表格会被当成图区域。实测力源 p144 的商标明细表
    就被检成了「图」（734x203），接着会被送进多模态模型白跑一次，
    并生成一个与工单3 表格块重复的图像块。

    对**细长矩形**（表格的边框线）用**中点判定**：这类图元不是零面积，
    而是 321x0.4 这种「很长很薄」的形状 —— 它们压在表格 bbox 的边界上，
    与 bbox 的**重叠面积恒为 0**，用面积比永远判不出"在表格里"。
    实测力源 p144 的两张商标表就是这样整表被漏排的（每张表 100+ 个边框矩形里
    只有不到一半被面积比判中，剩下的足以聚成一个假的"图区域"）。
    """
    if not boxes:
        return False
    cx, cy = (r[0] + r[2]) / 2.0, (r[1] + r[3]) / 2.0
    w, h = r[2] - r[0], r[3] - r[1]
    a = w * h
    thin = min(w, h) <= 2.0        # 线状图元（表格边框、分隔线）
    for b in boxes:
        if not (b[0] - tol <= cx <= b[2] + tol and b[1] - tol <= cy <= b[3] + tol):
            continue
        if thin or a <= 1.0:
            return True
        if _overlap_area(r, b) / a >= ratio:
            return True
    return False


def _collect_elements(page, pw: float, ph: float,
                      exclude_boxes: list | None = None) -> list[tuple]:
    """采集候选图元：位图 bbox ∪ 矢量绘图 bbox（都已裁到正文区、已排除表格区域）。

    ⚠️ 不要按 xref 去重。同一个 xref 会在页面上被**放置多次**
    （组织结构图的 29 个节点框只对应 8 个 xref），
    按 xref 去重会只保留第一次的矩形，把整张图拆散成十几个互不相连的小块，
    结果是整张图被 min_w/min_h 过滤掉 —— p39 实测就是这样"一个图都没检出来"。
    """
    elems: list[tuple] = []
    excl = list(exclude_boxes or [])

    for im in page.get_images(full=True):
        for r in page.get_image_rects(im[0]):
            c = _clip_to_content(_rect_of(page, r), pw, ph)
            if c and not _inside_any(c, excl):
                elems.append((c, "raster"))

    for d in page.get_drawings():
        # 纯装饰的 0.1pt 线段也要保留：它们是组织结构图各节点之间唯一的"桥"，
        # 没有它们，节点框之间的 39~55pt 空隙无法跨越，聚类会碎成一地。
        c = _clip_to_content(_rect_of(page, d["rect"]), pw, ph)
        if c and not _inside_any(c, excl):
            elems.append((c, "vector"))

    return elems


def cluster_rects(elems: list[tuple], gap: float = GAP_PT) -> list[list[int]]:
    """union-find：两个图元各扩张 gap/2 后相交即归为同一张图。

    **必须两边同时扩张**。只扩张一边等价于允许的间距只有 gap/2，
    实测会让 p38 的股权结构图（上下两块间距 23.3pt）被切成两张图。
    """
    n = len(elems)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    half = gap / 2.0
    expanded = [_expand(e[0], half) for e in elems]
    for i in range(n):
        ei = expanded[i]
        for j in range(i + 1, n):
            if _intersect(ei, expanded[j]):
                union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def _union_bbox(elems, idxs):
    xs0 = min(elems[i][0][0] for i in idxs)
    ys0 = min(elems[i][0][1] for i in idxs)
    xs1 = max(elems[i][0][2] for i in idxs)
    ys1 = max(elems[i][0][3] for i in idxs)
    return (xs0, ys0, xs1, ys1)


# ------------------------------------------------------------------ 图题
def _text_lines(page, pw: float, ph: float) -> list[tuple[tuple, str]]:
    """取页面上带 bbox 的文本行，**只保留正文区内的**。

    必须裁掉页眉/页码区，否则图题识别会把页眉
    「武汉力源信息技术股份有限公司 招股说明书」当成图题（它离图区域顶部只有 20pt 左右，
    完全落在 CAPTION_GAP_PT 之内），实测污染了 p38/p71/p110/p113 四页的图题。
    """
    out = []
    for blk in page.get_text("dict")["blocks"]:
        if blk.get("type") != 0:
            continue
        for ln in blk.get("lines", []):
            txt = "".join(sp.get("text", "") for sp in ln.get("spans", [])).strip()
            if not txt:
                continue
            r = _rect_of(page, ln["bbox"])
            # 整行都在页眉/页码区内的直接排除
            if r[3] <= MARGIN_TOP or r[1] >= ph - MARGIN_BOTTOM:
                continue
            out.append((r, txt))
    return out


# 章节标题/编号：这些东西紧贴图上方或下方，但它们**是正文的标题，不是图题**。
# 力源 p110 的业务流程图下方就是「(3)渠道销售」，p113 是「(4)大客户销售」；
# 若不排除，这两页的**章节标题会被当成图题从正文里删掉**。
_HEADING_RE = re.compile(r"^\s*(?:[（(]\s*[一二三四五六七八九十\d]+\s*[)）]"
                         r"|第[一二三四五六七八九十百]+[节章]"
                         r"|\d+(?:\.\d+)*\s*[、.．]"
                         r"|[一二三四五六七八九十]+\s*、)")
# 段落散文：以句末标点收尾的长句不是图题（p72 图上方那句
# 「在所有应用行业中位列第二，仅次于汽车电子。」就属于这类）。
_PROSE_END_RE = re.compile(r"[。；;]$")


def find_caption_lines(page, bbox, lines) -> list[str]:
    """图题 = 紧贴图区域上下 CAPTION_GAP_PT 内、水平有重叠、且**不是正文标题/段落**的文本行。

    返回**逐行**结果，不做拼接。原因：caption 用于从正文里整行剔除，
    而 pypdf 与 pymupdf 的行切分口径不同（一个是「2008年中国 IC 市场应用结构与增长(亿元)」，
    另一个是「2008 年中国IC 市场应用结构与增长(亿元)」），
    把多行拼成一个字符串再比对永远不会命中。逐行 + 归一化去空白才可靠。
    """
    x0, y0, x1, y1 = bbox
    cands = []
    for (lx0, ly0, lx1, ly1), txt in lines:
        hw = min(x1, lx1) - max(x0, lx0)
        if hw <= 1.0:
            continue
        # 水平重叠要达到较窄一方的 30% 以上，避免把左右栏文字误认成图题
        if hw < 0.30 * min(x1 - x0, lx1 - lx0):
            continue
        above = 0 <= (y0 - ly1) <= CAPTION_GAP_PT
        below = 0 <= (ly0 - y1) <= CAPTION_GAP_PT
        if not (above or below):
            continue
        t = txt.strip()
        if not t or _HEADING_RE.match(t) or _PROSE_END_RE.search(t):
            continue
        cands.append(((ly0, lx0), t))
    cands.sort()
    return [t for _, t in cands]


def find_caption(page, bbox, lines) -> str:
    """图题的可读形式（仅供展示/诊断）。正文剔除请用 find_caption_lines 的逐行结果。"""
    return " ".join(find_caption_lines(page, bbox, lines)).strip()


# ------------------------------------------------------------------ 主入口
def detect_page_figures(page, page_no: int,
                        gap: float = GAP_PT,
                        min_w: float = MIN_FIG_W,
                        min_h: float = MIN_FIG_H,
                        max_area_ratio: float = MAX_FIG_AREA,
                        table_boxes: list | None = None) -> PageFigures:
    """检测一页里的所有图区域（不裁剪）。

    table_boxes：该页的表格 bbox 列表（来自工单3 的 table_parser）。
    **必须传**，否则表格线框会被当成图。
    """
    pw, ph = float(page.rect.width), float(page.rect.height)
    elems = _collect_elements(page, pw, ph, exclude_boxes=table_boxes)
    pf = PageFigures(page=page_no)
    if not elems:
        return pf

    lines = _text_lines(page, pw, ph)
    groups = cluster_rects(elems, gap=gap)
    page_area = pw * ph

    kept = []
    for idxs in groups:
        bb = _union_bbox(elems, idxs)
        w, h = bb[2] - bb[0], bb[3] - bb[1]
        if w < min_w or h < min_h:
            continue
        if (w * h) / page_area > max_area_ratio:
            continue
        kinds = {elems[i][1] for i in idxs}
        kind = "mixed" if len(kinds) > 1 else kinds.pop()
        kept.append((bb, kind, len(idxs)))

    kept.sort(key=lambda t: (round(t[0][1]), round(t[0][0])))

    inner: dict[str, int] = {}
    for i, (bb, kind, n) in enumerate(kept, 1):
        fig = FigureRegion(page=page_no, index=i,
                           bbox=(round(bb[0], 1), round(bb[1], 1), round(bb[2], 1), round(bb[3], 1)),
                           kind=kind, n_elems=n)
        fig.caption = find_caption(page, bb, lines)
        pf.figures.append(fig)

        for cl in find_caption_lines(page, bb, lines):
            pf.caption_lines.add(norm_line(cl))

        # 收集图内文字的字符多重集（用于正文逐行剔除）
        for wd in page.get_text("words"):
            wr = (wd[0], wd[1], wd[2], wd[3])
            wa = (wr[2] - wr[0]) * (wr[3] - wr[1])
            if wa <= 0:
                continue
            if _overlap_area(wr, bb) / wa >= 0.5:
                for ch in norm_line(wd[4]):
                    inner[ch] = inner.get(ch, 0) + 1

    # 页级共享一个字符池：多张图共用一个池子。
    # 正文行只要**完全由图内字符构成**即判定为"图里漏出来的文字"并剔除。
    # 用多重集而非集合：图中有 6 个「销售处」、1 个「销售部」，
    # 计数正确才不会被跨行复用（例如把「销售处销售处」误判成正文）。
    pf.inner_chars = inner
    return pf


def line_is_in_figure(line: str, pf: PageFigures, max_len: int | None = None) -> bool:
    """判断一行正文是否该被剔除（因为它是"图里漏出来的文字"）。

    两条判据（满足其一即剔除）：
      A. **图题行**：归一化后与图题/资料来源行一致。
         这正是工单3「坑 13」的复现 —— p310 的图题
         「2008年中国IC市场应用结构与增长(亿元)」与 id=6 的问题字面重合度极高，
         若留在正文，它会形成一个「有标题、无数据」的碎片块并排到第一，
         把真正的图像块挤出候选。
      B. **图内文字**：归一化后非空，且其**每一个字符**都能在"图内字符多重集"
         里找到足够份数。力源 p39 的竖排节点名被逐字打散成
         「财」「务部」「销」「售」「部」，行字符串匹配既不可靠也不必要 ——
         字符级的多重集包含判定与解析器切分方式无关。
    """
    n = norm_line(line)
    if not n:
        return False
    if n in pf.caption_lines:
        return True
    if max_len is not None and len(n) > max_len:
        return False
    pool = dict(pf.inner_chars)
    for ch in n:
        if pool.get(ch, 0) <= 0:
            return False
        pool[ch] -= 1
    return True


def drop_lines_for_page(pf: PageFigures, lines: list[str], max_len: int = 60) -> set[str]:
    """给出一组需要剔除的正文行（原始字符串集合，供 pdf_parser 使用）。"""
    return {ln for ln in lines if line_is_in_figure(ln, pf, max_len=max_len)}


def crop_figure(page, fig: FigureRegion, out_path: Path, dpi: int = 150,
                pad: float = CROP_PAD_PT) -> FigureRegion:
    """把图区域渲染成 PNG。用页面渲染 + clip，位图与矢量图都能覆盖。"""
    import pymupdf

    pw, ph = float(page.rect.width), float(page.rect.height)
    x0 = max(0.0, fig.bbox[0] - pad)
    y0 = max(0.0, fig.bbox[1] - pad)
    x1 = min(pw, fig.bbox[2] + pad)
    y1 = min(ph, fig.bbox[3] + pad)
    clip = pymupdf.Rect(x0, y0, x1, y1)
    pix = page.get_pixmap(clip=clip, dpi=dpi)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pix.save(str(out_path))
    fig.raster_path = str(out_path)
    fig.px_w, fig.px_h = pix.width, pix.height
    return fig
