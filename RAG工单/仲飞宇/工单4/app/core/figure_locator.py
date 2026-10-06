# 工单编号：人工智能NLP-RAG-图像内容解析及检索优化
# 工单04 - 图像内容解析及检索优化
# 工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
# 工单03 - PDF文档的表格解析及检索优化
"""
图区定位：从一页 PDF 里找出"哪里是图"，交给多模态模型转写。

【为什么单独一个模块】定位逻辑有两个完全不同的来源（位图 / 矢量图），
各有自己的阈值和噪声，塞进 `pdf_parser.parse_page` 会把那个函数撑爆；
而且定位是**纯几何+文本**判断，可以脱离 Milvus 和 VLM 单独测。

【噪声有多严重 —— 这是本模块存在的全部理由】
实测《招股说明书2.pdf》350 页里 `get_image_info()` 返回 **5920 个图像实例**，
按本模块的规则过滤后只剩约 104 个 —— **98.2% 是噪声**。噪声有三种签名：

    1x394 / 1x914 / 1x1304 的 1 像素细条  → 被拉伸成窄矩形，是竖排文字或边框
    xref=2338 的 143x127 水印            → **每页重复 15 次**，铺满整页
    小 logo/图标                          → 面积占比很小
    整页扫描件（签字页/封面/附录）        → 面积占比 >60%，**它不是"插图"**，
                                          而且文字层已有内容，无须 VLM 转写

不滤掉它们，VLM 会把 5900 次调用浪费在分隔条和水印上。

【矢量图为什么不能靠"图元数量多"判断】
实测书1 有 319 页、书2 有 138 页的 `get_drawings()` 图元数 ≥15 —— 因为
**招股书满页都是表格，表格框线也是 drawing**。所以"图元多"判出来的是表格。
真正能区分的是下面这对条件（在 898 页上验证过，零误判）：

    ① 本页没有行数 ≥8 的表格
    ② **块级** bbox 满足 `宽≤22 且 26≤高≤120 且含≥2 汉字且不含数字` 的块 ≥5 个

命中：书2 页 37、38；书1 页 59 —— 全是真的组织结构图/流程图。

  · **必须用块级 bbox，不能用 span 级**：竖排标签在 span 级被逐字拆开
    （实测页 38 在 span 级命中 0、块级命中 17）。
  · **必须加条件 ①**：书2 页 207–218 是旋转 90° 的合并股东权益变动表，
    它的文本块天然"窄而高"，没有①会把 8 页财报全判成组织结构图。
  · **高度上限 ≤120 也必需**：组织结构图的标签是**一个词**（h=31.5~84），
    旋转页的"标签"是**整行**（h=690~764）。
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from app.config import settings

# 框内标签的汉字判定
_CJK_RE = re.compile(r"[一-鿿]")
_DIGIT_RE = re.compile(r"\d")


@dataclass
class Figure:
    """一个图区：几何 + 锚点 + （由 VLM 阶段回填的）语义文本。"""

    page_no: int                   # PDF position index（0 基）
    page_label: str                # 印刷页码
    bbox: tuple[float, float, float, float]
    kind: str                      # "chart"（图表）| "diagram"（结构图/示意图）
    title: str = ""                # 题注
    labels: str = ""               # 图内文本层文字（仅 diagram，按阅读序拼）
    # 合并前的原始 bbox，留痕用 —— 页 71 的两张位图合并成一个图区，
    # 验收时要能看出"确实是两块并成一块"，而不是只框住了其中一块
    src_bboxes: list[tuple[float, float, float, float]] = field(default_factory=list)
    # ---------- 以下由图像转写阶段回填 ----------
    text: str = ""                 # VLM 转写结果（或人工核对后的覆写）
    reviewed: bool = False         # 是否经过人工核对（见 image_cache.reviewed_text）
    vlm_seconds: float = 0.0       # 单图转写耗时，留痕用


@dataclass
class LocateStats:
    """定位统计，供入库脚本打印与验收留痕。"""

    n_figures: int = 0
    n_bitmap: int = 0
    n_vector: int = 0
    n_images_dropped: int = 0      # 被四道规则滤掉的位图实例数


# ----------------------------------------------------------------------
# 几何工具
# ----------------------------------------------------------------------
def _overlap_ratio(a0: float, a1: float, b0: float, b1: float) -> float:
    """一维区间重叠占**较短那个**的比例。"""
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    shorter = min(a1 - a0, b1 - b0)
    return inter / shorter if shorter > 1e-6 else 0.0


def _mergeable(a, b) -> bool:
    """两个 bbox 是否应该并成一个图区。

    【为什么取"宁合不拆"】页 71 的图是**饼图 + 柱状图并排**（间隙 0），
    合并规则太紧就会只转写饼图、丢掉柱状图 —— 而"负增长的是哪个行业"
    恰恰只在柱状图上（id=6 直接挂）。合并太松的代价只是把相邻的表也框进来，
    VLM 多转写一段已有内容，浪费一个 top-k 名额而已，不致命。
    """
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    xov = _overlap_ratio(ax0, ax1, bx0, bx1)
    yov = _overlap_ratio(ay0, ay1, by0, by1)
    h_gap = max(0.0, max(ax0, bx0) - min(ax1, bx1))
    v_gap = max(0.0, max(ay0, by0) - min(ay1, by1))
    if yov > 0.6 and h_gap <= 30:          # 并排面板
        return True
    if xov > 0.6 and v_gap <= 20:          # 上下堆叠
        return True
    return False


def _union(boxes) -> tuple[float, float, float, float]:
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))


def merge_boxes(boxes: list[tuple[float, float, float, float]],
                pad: float = 4.0) -> list[tuple[float, float, float, float]]:
    """把互相邻近的框并成图区，迭代到不动点，最后外扩 pad。"""
    cur = [tuple(float(v) for v in b) for b in boxes]
    changed = True
    while changed and len(cur) > 1:
        changed = False
        out: list[tuple[float, float, float, float]] = []
        while cur:
            a = cur.pop()
            for i, b in enumerate(cur):
                if _mergeable(a, b):
                    cur[i] = _union([a, b])
                    changed = True
                    break
            else:
                out.append(a)
        cur = out
    return [(b[0] - pad, b[1] - pad, b[2] + pad, b[3] + pad) for b in cur]


# ----------------------------------------------------------------------
# 位图
# ----------------------------------------------------------------------
def scan_xref_pages(doc, n_pages: int) -> dict[int, int]:
    """预扫描：每个图像 xref 出现在多少页上。

    【为什么要预扫描】水印的判据之一是"同一个 xref 跨多页重复出现"。
    单页解析时看不到全局，所以先花一次轻量扫描（只取图像清单，不抽文本）
    把跨页计数建好。实测这一步在 898 页上很快（毫秒级/页）。
    """
    counts: Counter[int] = Counter()
    for i in range(n_pages):
        seen = {im[0] for im in doc[i].get_images(full=True)}
        counts.update(seen)
    return dict(counts)


def bitmap_boxes(page, xref_page_count: dict[int, int],
                 stats: LocateStats | None = None
                 ) -> list[tuple[float, float, float, float]]:
    """页面上"像图"的位图 bbox（四道过滤，见模块 docstring）。"""
    infos = page.get_image_info(xrefs=True)
    per_page = Counter(i.get("xref", 0) for i in infos)
    page_area = page.rect.width * page.rect.height
    kept = []
    for info in infos:
        xref = info.get("xref", 0)
        w, h = info.get("width", 0), info.get("height", 0)
        x0, y0, x1, y1 = info["bbox"]
        area = max(0.0, x1 - x0) * max(0.0, y1 - y0)

        drop = (
            min(w, h) < settings.image_min_bitmap_px                  # R1 细条
            or per_page[xref] >= settings.image_max_repeat_per_page    # R2 本页水印平铺
            or xref_page_count.get(xref, 0) >= settings.image_max_repeat_pages  # R3 跨页水印
            or area < settings.image_min_area_ratio * page_area         # R4 小 logo
            or area > settings.image_max_area_ratio * page_area         # R5 整页扫描件
        )
        if drop:
            if stats is not None:
                stats.n_images_dropped += 1
            continue
        kept.append((float(x0), float(y0), float(x1), float(y1)))
    return kept


# ----------------------------------------------------------------------
# 矢量图（组织结构图 / 流程图）
# ----------------------------------------------------------------------
def _is_vertical_label(text: str, bbox) -> bool:
    """是不是组织结构图里那种"竖排的一个词"标签。

    实测页 38 的标签：宽 10.5、高 31.5~84、内容是完整词（'销售部'、'大客户销售部'、
    '电话及网络销售部'）。注意高度上限：旋转财报页（207–218）那些"窄而高"的块
    高 690~764，是整行不是词。
    """
    x0, y0, x1, y1 = bbox
    w, h = x1 - x0, y1 - y0
    if not (w <= 22 and 26 <= h <= 120):
        return False
    if len(_CJK_RE.findall(text)) < 2:
        return False
    return not _DIGIT_RE.search(text)


def _vector_region(page, texts, tables):
    """矢量图区（organizational chart 那类）。没有则返回 None。"""
    if not settings.enable_vector_figures:
        return None, ""
    # 条件①：本页有"正经表格"就认为图元是表格框线，不判图
    if any(len(t.rows) >= 8 for t in tables):
        return None, ""
    # 条件②：块级竖排标签 ≥5 个
    labels = [t for t in texts if _is_vertical_label(t.text, t.bbox)]
    if len(labels) < 5:
        return None, ""

    labels_box = _union([t.bbox for t in labels])
    page_area = page.rect.width * page.rect.height
    drawings = page.get_drawings()
    # 外框：面积够大、且把标签框大部分包住的那个矩形
    frame = None
    best_area = 0.0
    for g in drawings:
        r = g["rect"]
        if r.get_area() < 0.15 * page_area:
            continue
        if not _overlap_ratio(r.y0, r.y1, labels_box[1], labels_box[3]) > 0.6:
            continue
        if not _overlap_ratio(r.x0, r.x1, labels_box[0], labels_box[2]) > 0.6:
            continue
        if r.get_area() > best_area:
            frame, best_area = r, r.get_area()
    region = ((frame.x0, frame.y0, frame.x1, frame.y1) if frame is not None
              else labels_box)
    # 把落在区内的、尺寸合理的矩形也并进来（框线本身在图外沿时）
    extra = []
    for g in drawings:
        r = g["rect"]
        if r.get_area() <= 0 or min(r.width, r.height) < 12 or max(r.width, r.height) > 400:
            continue
        if (_overlap_ratio(r.y0, r.y1, region[1], region[3]) > 0.3
                and _overlap_ratio(r.x0, r.x1, region[0], region[2]) > 0.3):
            extra.append((r.x0, r.y0, r.x1, r.y1))
    if extra:
        region = _union([region] + extra)
    # 图内文字按阅读序（先上后下、同高先左后右）
    ordered = sorted(labels, key=lambda t: (round(t.bbox[1] / 12), t.bbox[0]))
    return region, "、".join(t.text for t in ordered)


# ----------------------------------------------------------------------
# 题注
# ----------------------------------------------------------------------
def _caption_ok(text: str) -> bool:
    """像不像题注。"""
    if not 4 <= len(text) <= 60:
        return False
    if len(_CJK_RE.findall(text)) < 2:
        return False
    if text.endswith(("。", "；", "：", "，", "%")):
        return False
    return not _DIGIT_RE.fullmatch(text)


def caption_for(bbox, texts, prev_texts, page_height: float) -> str:
    """图区上方最近的题注；本页找不到且图在页顶时，回退到上一页末尾。

    【为什么要回退上一页】实测组织结构图的题注在**页 37**、图体在**页 38** ——
    印刷上是一张跨页的图。只查本页会得到空题注，图块就少了最重要的语义锚点。
    """
    x0, y0, x1, y1 = bbox
    cands = [t for t in texts
             if t.bbox[3] <= y0 and y0 - t.bbox[3] <= settings.image_caption_gap
             and _caption_ok(t.text)]
    if not cands and y0 < page_height * 0.15 and prev_texts:
        cands = [t for t in prev_texts[-3:] if _caption_ok(t.text)]
    if not cands:
        return ""
    return max(cands, key=lambda t: t.bbox[3]).text


# ----------------------------------------------------------------------
# 主入口
# ----------------------------------------------------------------------
def locate_figures(page, texts, tables, *,
                   page_no: int, page_label: str,
                   xref_page_count: dict[int, int],
                   prev_texts=None,
                   stats: LocateStats | None = None) -> list[Figure]:
    """找出本页所有图区。

    位图与矢量两条来源各自独立判断，结果**不合并**（一张页面上可能既有位图图表
    又有矢量结构图，页 37/38 就是这种情况）。每页上限见
    `settings.image_max_figures_per_page`。
    """
    out: list[Figure] = []
    prev_texts = prev_texts or []

    # ---- 位图 ----
    boxes = bitmap_boxes(page, xref_page_count, stats)
    page_area = page.rect.width * page.rect.height
    for box in merge_boxes(boxes):
        # R5 要在**合并之后**再判一次：书2 的签字页是 8 条位图拼成的
        # （单条只占 11% 页面积，合起来才是整页 92%），只看单条会漏掉它们。
        if (box[2] - box[0]) * (box[3] - box[1]) > settings.image_max_area_ratio * page_area:
            if stats is not None:
                stats.n_images_dropped += 1
            continue
        if stats is not None:
            stats.n_bitmap += 1
        out.append(Figure(
            page_no=page_no, page_label=page_label,
            bbox=box, kind="chart",
            title=caption_for(box, texts, prev_texts, page.rect.height),
            src_bboxes=[b for b in boxes
                        if _overlap_ratio(b[1], b[3], box[1], box[3]) > 0.5],
        ))

    # ---- 矢量图 ----
    region, labels = _vector_region(page, texts, tables)
    if region is not None:
        if stats is not None:
            stats.n_vector += 1
        out.append(Figure(
            page_no=page_no, page_label=page_label,
            bbox=region, kind="diagram",
            title=caption_for(region, texts, prev_texts, page.rect.height),
            labels=labels, src_bboxes=[region],
        ))

    if len(out) > settings.image_max_figures_per_page:
        # 太多就并成一个（页 115/117 有几十个截图拼版）
        merged = merge_boxes([f.bbox for f in out], pad=0.0)
        kind = out[0].kind
        out = [Figure(page_no=page_no, page_label=page_label, bbox=b, kind=kind,
                            title=caption_for(b, texts, prev_texts, page.rect.height),
                            labels="", src_bboxes=[f.bbox for f in out]) for b in merged]

    if stats is not None:
        stats.n_figures += len(out)
    return out
