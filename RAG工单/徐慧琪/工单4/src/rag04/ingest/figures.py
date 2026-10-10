# 工单编号：人工智能NLP-RAG-图像内容解析及检索优化
"""★图区检测：矢量图形聚类 + 题注锚定 + 整页兜底。

设计硬约束（实测得出）：绝不静默丢答案。
裁切框过短时多模态模型不会报错，而是自信地给出错误答案。
因此本模块宁多给像素，绝不裁短；置信度不足时降级整页渲染。
"""
from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import fitz

from rag04.config import Settings
from rag04.schema import FigureBlock

logger = logging.getLogger("rag04.figures")

# 图题注与图表提示语（find_captions 用的**宽**匹配，供题注锚定路径使用）。
# 编号部分收紧为 `图\s?\d+` 且「图」前不得紧邻汉字，否则公司名「华创兴图」
# 加空白/数字就会被当成「图N」：实测 p66「华创兴图  2012-05-09」、
# p214「华创兴图19件」均由此误报。注意宽匹配仍保留 增长率/产业链 等词——
# 它们只在**证据闸门**（looks_like_caption）里被剔除，锚定路径依赖宽召回。
_CAPTION_PAT = re.compile(
    r"((?<![一-鿿])图\s?\d+(?:[-－.]\d+)*\s*[:：]?"   # 图 1-1 / 图 2：
    r"|如下图|如下图[:：]?|下图"
    r"|组织结构图|组织架构图|股权结构图"
    r"|应用结构|增长图|增长率|产业链|示意图|流程图|柱状图|饼图)"
)

# --- 整页兜底的严格证据闸门（looks_like_caption / has_figure_evidence）---

# 题注是一行标题，不是句子；超过该长度一律视为正文。实测 p41
# 「2016 年至2018 年，公司营业收入的复合增长率为57.82%，…」即因分块
# 恰好较短而险些过关，最终靠关键词剔除拦住。
_CAPTION_MAX_CHARS = 80
_SENTENCE_END = ("。", "！", "？", "；")   # 供 str.endswith 使用的元组

# 明确的图表标题关键词。刻意剔除本语料正文高频词：
#   * 增长率 —— 实测命中 p11 表头「金额  增长率  金额」与 p41 正文句；
#   * 产业链 —— 实测命中 p9/p47 风险提示套话；
#   * 应用结构 —— 仅在题注形状（短 + 无句末标点）下承认，以保住 p72
#     真题注「2008 年中国IC 市场应用结构与增长(亿元)」；该词全语料仅此一处。
_TITLE_KEYWORDS = (
    "组织结构图", "组织架构图", "股权结构图", "结构图", "示意图", "流程图",
    "柱状图", "饼图", "增长图", "应用结构", "框架图", "路线图", "趋势图",
    "分布图", "对比图", "拓扑图", "曲线图", "甘特图",
)

# 图/表编号标签必须位于块首（允许前置空白）。仅收紧空白（图\s?\d+）不够：
# 「兴图19件」的「图19」仍会命中，故锚定块首。
_CAPTION_LABEL_PAT = re.compile(r"^\s*(?:图|表)\s?\d+(?:[-－.]\d+)*")

# 表格网格特征：大量短直线 → 疑似表格，避免把表格识别为图
_MAX_LINES_FOR_FIGURE = 400

# 真实图表位图的最小像素面积。水印 logo 为 143x127≈1.8 万像素且整页平铺 15 次；
# 真实图表为 638x479≈30.6 万、525x473≈24.8 万像素。10 万可干净分离。
_MIN_FIGURE_PIXELS = 100_000


@dataclass
class GroundTruthBox:
    doc_id: str
    page: int
    must_include: tuple[float, float, float, float]


def _bbox_union(rects: list[fitz.Rect]) -> fitz.Rect:
    x0 = min(r.x0 for r in rects); y0 = min(r.y0 for r in rects)
    x1 = max(r.x1 for r in rects); y1 = max(r.y1 for r in rects)
    return fitz.Rect(x0, y0, x1, y1)


def _collect_drawing_rects(page: fitz.Page) -> list[fitz.Rect]:
    """收集有意义的绘图矩形，过滤掉装饰性与过小元素。"""
    out: list[fitz.Rect] = []
    try:
        drawings = page.get_drawings()
    except Exception as e:
        logger.warning("get_drawings 失败，将走兜底路径：%s", e)
        return out

    for d in drawings:
        r = d.get("rect")
        if r is None:
            continue
        rr = fitz.Rect(r)
        if rr.get_area() < 4.0:        # 忽略极小装饰
            continue
        if rr.width < 2.0 or rr.height < 2.0:
            continue
        out.append(rr)
    return out


def collect_image_rects(page: fitz.Page) -> list[fitz.Rect]:
    """收集**嵌入位图**形式的图表区域。

    为什么需要这条路：本语料的目标图表有两种来源——
      * 组织结构图（第 39 页）是**矢量绘制**，`get_image_info()` 只返回水印与细条；
      * IC 市场图（第 72 页）是**两张真实嵌入位图**，矢量聚类完全找不到它。
    只做矢量聚类会漏掉 id 6，只做位图会漏掉 id 5，两条都必须走。

    滤除干扰：水印 logo 为 143x127（约 1.8 万像素）且整页平铺 15 次，
    组织结构图页还有一片 1px 宽的渐变细条。用像素面积阈值可一并滤除。
    """
    out: list[fitz.Rect] = []
    try:
        infos = page.get_image_info()
    except Exception as e:
        logger.warning("get_image_info 失败，跳过位图来源：%s", e)
        return out

    for info in infos:
        w = int(info.get("width", 0) or 0)
        h = int(info.get("height", 0) or 0)
        if w * h < _MIN_FIGURE_PIXELS:
            continue
        b = info.get("bbox")
        if not b:
            continue
        r = fitz.Rect(b)
        if r.width < 20 or r.height < 20:
            continue
        out.append(r)
    return out


def cluster_rects(rects: list[fitz.Rect], gap: float = 12.0) -> list[fitz.Rect]:
    """并查集把间距小于 gap 的矩形聚成一簇，返回每簇的并集外框。"""
    n = len(rects)
    if n == 0:
        return []
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    inflated: list[fitz.Rect] = []
    for r in rects:
        ir = fitz.Rect(r)
        ir.x0 -= gap; ir.y0 -= gap; ir.x1 += gap; ir.y1 += gap
        inflated.append(ir)

    for i in range(n):
        for j in range(i + 1, n):
            if inflated[i].intersects(inflated[j]):
                union(i, j)

    groups: dict[int, list[fitz.Rect]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(rects[i])

    return [_bbox_union(g) for g in groups.values()]


def find_captions(page: fitz.Page) -> list[tuple[fitz.Rect, str]]:
    """返回 (bbox, 文本) 列表，文本命中图题注或图表提示语。"""
    out: list[tuple[fitz.Rect, str]] = []
    try:
        blocks = page.get_text("blocks")
    except Exception:
        return out
    for b in blocks:
        if len(b) < 5:
            continue
        x0, y0, x1, y1, text = b[0], b[1], b[2], b[3], str(b[4])
        t = text.strip()
        if not t:
            continue
        if _CAPTION_PAT.search(t):
            out.append((fitz.Rect(x0, y0, x1, y1), t.replace("\n", " ")[:200]))
    return out


def looks_like_caption(text: str) -> bool:
    """严格题注形状判定：整页兜底闸门专用，杜绝关键词误报。

    与 find_captions() 的宽匹配不同，这里要求**同时**满足：
      1. 短（<= _CAPTION_MAX_CHARS 字符）——题注是一行标题，不是句子；
      2. 不以句末标点（。！？；）结尾——分块截断的正文句常满足长度但仍带
         句末标点，或多在 80 字以上；
      3. 块首即图/表编号标签（图1-1、图 2：、表1-1），或含明确的图表标题
         关键词（示意图/流程图/组织结构图/应用结构…，见 _TITLE_KEYWORDS）。

    实测反例（修复前误当题注 → 伪造整页图）：
      * p11「金额  增长率  金额」——表头，关键词 增长率 已剔除；
      * p41「…公司营业收入的复合增长率为57.82%…」——正文，关键词已剔除；
      * p9/p47「…产业链…」风险提示套话——产业链 已剔除；
      * p66「华创兴图  2012-05-09」——编号标签必须块首，且「图」前不得为汉字。
    真题注 p72「2008 年中国IC 市场应用结构与增长(亿元)」由第 1/2/3 条
    （24 字、无句末标点、含 应用结构）共同承认。
    """
    t = " ".join(str(text).split())
    if not t or len(t) > _CAPTION_MAX_CHARS:
        return False
    if t.endswith(_SENTENCE_END):
        return False
    if _CAPTION_LABEL_PAT.match(t):
        return True
    return any(k in t for k in _TITLE_KEYWORDS)


def has_figure_evidence(page: fitz.Page, rects: list[fitz.Rect] | None = None) -> bool:
    """整页兜底前的正向证据闸门，两个独立条件必须同时成立。

    (a) 页内确有图形：复用调用方在 _candidates() 采集的绘图/嵌入位图矩形
        （`rects`），不重复调用 get_drawings/get_image_info。纯文本页、表格页
        的装饰性细线（<2pt）与 143x127 水印已在采集阶段滤除，故为空。
    (b) 至少一个文本块形如真题注：looks_like_caption() 的严格形状判定。

    仅凭关键词命中（旧行为）会把表头、正文句、风险提示套话当成题注，
    对这些页面整页兜底只会与文本块重复、污染检索并放大 VLM 成本。
    """
    if rects is None:
        rects = _collect_drawing_rects(page) + collect_image_rects(page)
    if not rects:
        return False
    try:
        blocks = page.get_text("blocks")
    except Exception:
        return False
    for b in blocks:
        if len(b) < 5:
            continue
        if looks_like_caption(str(b[4])):
            return True
    return False


def expand(rect: fitz.Rect, margin_ratio: float, page_rect: fitz.Rect) -> fitz.Rect:
    """按比例外扩，并夹紧到页面边界内。

    先与页面求交再外扩：完全落在页面外的候选直接返回空矩形（is_empty），
    由调用方跳过。若独立夹紧 x0 与 x1，页外候选会被翻转成空/反向矩形，
    `get_pixmap(clip=...)` 将直接崩溃。
    """
    base = fitz.Rect(rect) & fitz.Rect(page_rect)
    if base.is_empty:
        return fitz.Rect()
    dx = base.width * margin_ratio
    dy = base.height * margin_ratio
    out = fitz.Rect(base.x0 - dx, base.y0 - dy, base.x1 + dx, base.y1 + dy)
    out.x0 = max(out.x0, page_rect.x0)
    out.y0 = max(out.y0, page_rect.y0)
    out.x1 = min(out.x1, page_rect.x1)
    out.y1 = min(out.y1, page_rect.y1)
    return out


def _line_heavy(page: fitz.Page, rect: fitz.Rect) -> bool:
    """区域是否以短直线为主（疑似表格网格）。"""
    try:
        n = 0
        for d in page.get_drawings():
            r = d.get("rect")
            if r is None:
                continue
            rr = fitz.Rect(r)
            if rect.intersects(rr) and rr.width < 3.0 or (rect.intersects(rr) and rr.height < 3.0):
                n += 1
                if n > _MAX_LINES_FOR_FIGURE:
                    return True
        return False
    except Exception:
        return False


def _candidates(page: fitz.Page, s: Settings,
                rects: list[fitz.Rect] | None = None
                ) -> list[tuple[fitz.Rect, str, float]]:
    """产出 (rect, method, confidence) 候选。

    `rects` 允许调用方复用已采集的绘图/位图矩形，避免 detect_figures 为
    证据闸门再跑一遍 get_drawings/get_image_info（大文档上开销可观）。
    """
    # 两条来源缺一不可：矢量绘制（组织结构图）+ 嵌入位图（IC 市场图）
    if rects is None:
        rects = _collect_drawing_rects(page) + collect_image_rects(page)
    clusters = cluster_rects(rects, gap=s.fig_gap)
    # 只保留足够大的簇，并排除疑似表格
    big = [c for c in clusters
           if c.get_area() >= s.fig_min_area and not _line_heavy(page, c)]

    captions = find_captions(page)
    out: list[tuple[fitz.Rect, str, float]] = []

    # 策略1：题注锚定——取题注上方最近且水平重叠的簇，合并题注本身
    if captions:
        for cap_rect, _txt in captions:
            overlapping = [
                c for c in big
                if c.y1 <= cap_rect.y0 + (cap_rect.y0 - c.y0)  # 在题注上方或与题注相邻
                and not (c.x1 < cap_rect.x0 or c.x0 > cap_rect.x1)
            ]
            same_band = [c for c in big if abs(c.y0 - cap_rect.y0) < 60 or c.intersects(cap_rect)]
            pool = same_band or overlapping
            if pool:
                nearest = max(pool, key=lambda c: c.get_area())
                merged = _bbox_union([nearest, cap_rect])
                out.append((merged, "caption_anchor", 0.9))

    # 策略2：纯矢量聚类
    for c in big:
        out.append((c, "vector_cluster", 0.6))

    return out


def detect_figures(page: fitz.Page, doc_id: str, page_no: int,
                   s: Settings) -> list[FigureBlock]:
    """检测本页图区，返回已扩边的 FigureBlock 列表（尚未渲染）。"""
    page_rect = page.rect
    # 矢量+位图矩形只采集一次，候选定位与兜底证据闸门共用同一份结果。
    rects = _collect_drawing_rects(page) + collect_image_rects(page)
    cands = _candidates(page, s, rects=rects)

    # 策略3：整页兜底——必须同时具备两条**独立**证据：
    #   (a) 页内确有图形（rects 非空，装饰细线与水印已滤除）；
    #   (b) 至少一个文本块形如真题注（短、无句末标点、编号标签或标题关键词）。
    # 只看关键词的旧闸门会把表头/正文句/风险提示套话当成题注（实测 p9、p11、
    # p41、p47、p66），对这些页面整页渲染只会与文本块重复并放大 VLM 成本。
    if not cands:
        if has_figure_evidence(page, rects=rects):
            logger.info("p%d 有图区证据但无可靠候选，降级整页渲染", page_no)
            cands = [(fitz.Rect(page_rect), "full_page", 0.3)]
        else:
            logger.info("p%d 无图区证据，不产出图区", page_no)
            return []

    # 同法合并后去重
    merged: list[tuple[fitz.Rect, str, float]] = []
    for rect, method, conf in sorted(cands, key=lambda x: -x[2]):
        if any(rect.intersects(m[0]) and
               rect.get_area() > 0 and
               fitz.Rect(m[0]).intersects(rect) and
               (rect & fitz.Rect(m[0])).get_area() / max(rect.get_area(), 1e-6) > 0.7
               for m in merged):
            continue
        merged.append((rect, method, conf))

    figs: list[FigureBlock] = []
    for rect, method, conf in merged:
        final = expand(rect, s.crop_margin, page_rect)
        if final.is_empty:
            logger.warning("p%d 候选完全落在页面外，跳过：%s", page_no, rect)
            continue
        caption = ""
        caps = find_captions(page)
        if caps:
            inside = [t for r, t in caps if final.intersects(r)]
            caption = inside[0] if inside else ""
        figs.append(FigureBlock(
            doc_id=doc_id, page=page_no,
            bbox=(final.x0, final.y0, final.x1, final.y1),
            image_path="", caption=caption,
            detect_method=method, confidence=conf,
        ))
    return figs


def render_figure(page: fitz.Page, fig: FigureBlock, s: Settings) -> str:
    """按 bbox 高 DPI 渲染落盘，返回绝对路径（POSIX 风格）。

    bbox 为空/非法（空矩形、反向矩形、NaN、长度不对）时抛 ValueError。
    模块内路径已由 expand() 保证非空，但本函数是公开 API，外部直接构造的
    FigureBlock 不能在 get_pixmap(clip=...) 里以底层报错崩溃。
    """
    s.fig_cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        clip = fitz.Rect(*fig.bbox)
    except Exception as e:  # PyMuPDF 对参数个数错误抛基类 Exception
        raise ValueError(f"图区 bbox 非法，无法渲染：{fig.bbox!r}") from e
    if clip.is_empty or clip.is_infinite or not (clip.width > 0 and clip.height > 0):
        raise ValueError(f"图区 bbox 为空或无效，无法渲染：{tuple(clip)!r}")
    pix = page.get_pixmap(dpi=s.render_dpi, clip=clip)
    key = hashlib.md5(
        f"{fig.doc_id}|{fig.page}|{fig.bbox}|{s.render_dpi}".encode("utf-8")
    ).hexdigest()[:12]
    out = Path(s.fig_cache_dir) / f"{fig.doc_id}_p{fig.page}_{key}.png"
    pix.save(str(out))
    fig.image_path = out.as_posix()
    return fig.image_path
