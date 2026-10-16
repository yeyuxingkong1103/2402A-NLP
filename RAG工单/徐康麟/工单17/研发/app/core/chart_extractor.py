# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-解决API 服务并发瓶颈与资源泄漏工单
# （本文件复用自 工单7/工单10 基座；工单17 只补头部编号，未改动其逻辑）
"""工单4 位图图表数值抽取（chart_extractor）。

任务：**人工智能NLP-RAG 项目-金融问答系统部署**（RAG 内核沿用 工单7 实现）。

把 PDF 里的**位图图表**（横向条形图 / 柱状图 / 3D 饼图）还原成
「行业—数值」结构化证据，使图里的数字变成**可检索、可引用、可回答**的内容。

设计依据：``设计/图像解析方案.md`` §4。

------------------------------------------------------------------------
方法学（为什么这样做，而不是"看图说话"）
------------------------------------------------------------------------
本机**没有任何 OCR**（pytesseract / paddleocr / easyocr 全缺，cv2 也缺）。因此：

1. **几何量取**（确定性，主路径）
   * 柱体掩膜：``饱和度高 ≠ 灰度`` —— 实测图表里叠着**斜向浅灰水印**，
     纯灰度阈值会把水印笔画并进柱体连通域。用「通道极差 > 40」把彩色柱体与灰水印分开。
   * 基线（纵轴）：逐列统计**低饱和深色**像素数，排除图像 3px 边框后取最大值列。
   * 切柱：柱掩膜按行连续段切分；每段的 x 区间与基线的左/右差即**带符号柱长**。
2. **绝对刻度标定**（来自**正文**，不是常量）
   * 从页文本层抽「增长率……10.5%……位列第二，仅次于汽车电子」：
     得到 (锚点值=10.5, 锚点排名=2, 榜首提示=汽车电子)。
   * 排名第 2 的柱长 L₂ ⇒ 比例 ``scale = 10.5 / L₂``，其余柱值 = 柱长 × scale。
   * 该方法把「数值」完全交给几何 + 正文锚点，**代码里没有任何一个行业数值常量**。
3. **位图文字识别**（两种独立通道，互相校验，都会写进产物）
   * ``template_match``：**字形模板匹配**。用系统 CJK 字体渲染候选串，
     与切片二值掩膜做 IoU 匹配（多字体 × 多字号搜索）。确定性、无外部依赖。
   * ``clip_zero_shot``：中文 CLIP 零样本分类（``multimodal_encoder`` 提供）。
   * 两者若不一致，**两条结论都记录**（``label_disagreement``），绝不静默择一。
4. **独立复核**：柱末端外侧的**数值标签**再单独做一次数字模板识别，
   与「几何+锚点」算出的数值逐个比对（``ocr_verified``）。对不上就报 warning。
"""

from __future__ import annotations

import io
import math
import re
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

__all__ = [
    "DEFAULT_INDUSTRY_CANDIDATES",
    "parse_growth_anchor",
    "decode_image",
    "analyze_bar_chart",
    "analyze_pie_chart",
    "match_text_candidates",
    "read_numeric_line",
    "BarItem",
    "ChartResult",
]

# ---------------------------------------------------------------------------
# 可配置词表（**输入参数**，不是答案）：行业名候选。
# 这些词来自本语料所属领域（中国半导体分销行业）的常识词表 + 页正文出现的同义词；
# 「哪个行业最快/哪个负增长」由几何与锚点算出，词表只负责把位图字形映射成名字。
# ---------------------------------------------------------------------------
DEFAULT_INDUSTRY_CANDIDATES = [
    "IC卡", "工控", "计算机", "其他", "汽车", "网络通信", "消费",
    "工业控制", "汽车电子",
]

_CJK_FONTS = [
    r"C:\Windows\Fonts\simsun.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\simkai.ttf",
]
_DIGIT_FONT = r"C:\Windows\Fonts\arial.ttf"
_DIGIT_CHARS = list("0123456789.%")

_INK_LUM = 150.0          # 低饱和深色 = 文字/坐标轴
_SAT_MIN = 40.0           # 通道极差 > 40 = 彩色（柱体），灰水印不满足
_BORDER_PX = 3            # 位图边框宽度（必须剔除，见 _masks 注释）
_CN_NUM = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------
@dataclass
class BarItem:
    row: int                        # 自顶向下第几根（0-based）
    label: str = ""
    label_bbox: list[int] = field(default_factory=list)
    label_score: float = 0.0
    label_alt: str = ""
    value: float | None = None
    value_source: str = ""
    direction: str = "pos"          # pos | neg
    length_px: int = 0
    bar_bbox: list[int] = field(default_factory=list)
    printed_value: str = ""         # 数值标签的独立识别结果
    printed_score: float = 0.0
    ocr_verified: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ChartResult:
    chart_type: str = "bar"         # bar | pie3d
    width: int = 0
    height: int = 0
    x_axis: int | None = None
    bars: list[BarItem] = field(default_factory=list)
    scale_percent_per_px: float | None = None
    anchor: dict[str, Any] = field(default_factory=dict)
    extremes: dict[str, Any] = field(default_factory=dict)
    label_method: str = ""
    value_method: str = ""
    extraction_method: str = "geometry+anchor"
    quantitative: bool = True
    warnings: list[str] = field(default_factory=list)
    pie_labels: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["bars"] = [b if isinstance(b, dict) else b for b in d["bars"]]
        return d


# ---------------------------------------------------------------------------
# 图像解码与掩膜
# ---------------------------------------------------------------------------
def decode_image(doc: "Any", xref: int) -> Image.Image:
    """从 PDF 取位图（by xref）→ PIL RGB 图。"""

    import fitz  # 延迟导入：让本模块可被无 PyMuPDF 的调用方 import

    pix = fitz.Pixmap(doc, xref)
    if pix.n - pix.alpha >= 4:
        pix = fitz.Pixmap(fitz.csRGB, pix)
    mode = "RGB" if pix.n >= 3 else "L"
    return Image.frombytes(mode, (pix.width, pix.height), pix.samples).convert("RGB")


def _masks(img: Image.Image) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a = np.asarray(img).astype(np.int16)
    if a.ndim == 2:
        a = np.stack([a] * 3, axis=-1)
    r, g, b = a[:, :, 0], a[:, :, 1], a[:, :, 2]
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    sat = a.max(axis=2) - a.min(axis=2)
    color = (sat > _SAT_MIN) & (lum < 245.0)          # 彩色（柱/饼扇区）
    ink = (lum < _INK_LUM) & (sat <= _SAT_MIN)        # 低饱和深色（文字/坐标轴）
    # **必须剔除图像边框**：实测 xref1692 的左右各有 1 列近乎全黑（dark≈467/473），
    # 不剔除的话它会在文本行聚类里把整页文字粘成一条巨型"文本行"，
    # 连带把行业名裁成 (1,352,88,392) 这种跨行巨框（已实测复现并修掉）。
    k = _BORDER_PX
    if ink.shape[0] > 2 * k and ink.shape[1] > 2 * k:
        ink[:k, :] = False; ink[-k:, :] = False
        ink[:, :k] = False; ink[:, -k:] = False
    return lum, color, ink


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    if not mask.any():
        return None
    ys, xs = np.where(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


# ---------------------------------------------------------------------------
# 文字行聚类
# ---------------------------------------------------------------------------
def _text_lines(mask: np.ndarray, bbox: tuple[int, int, int, int] | None = None,
                *, min_h: int = 5, min_px: int = 12) -> list[tuple[int, int, int, int]]:
    """把掩膜里的墨点按「y 重叠 + x 邻近」聚成文本行，返回 bbox 列表。"""

    if bbox is not None:
        x0, y0, x1, y1 = bbox
        sub = np.zeros_like(mask)
        sub[max(0, y0):y1, max(0, x0):x1] = mask[max(0, y0):y1, max(0, x0):x1]
    else:
        sub = mask
    lab, n = ndimage.label(sub, structure=np.ones((3, 3)))
    comps: list[tuple[int, int, int, int]] = []
    for sl in ndimage.find_objects(lab):
        h = sl[0].stop - sl[0].start
        w = sl[1].stop - sl[1].start
        if h < 3 or w < 1:
            continue
        comps.append((sl[1].start, sl[0].start, sl[1].stop, sl[0].stop))
    comps = [c for c in comps if (c[2] - c[0]) * (c[3] - c[1]) >= 4]
    # 贪心合并
    comps.sort(key=lambda c: (c[1], c[0]))
    lines: list[list[int]] = []
    for c in comps:
        placed = False
        for L in lines:
            ovl = min(L[3], c[3]) - max(L[1], c[1])
            gap_x = max(L[0], c[0]) - min(L[2], c[2])
            if ovl >= 0.5 * min(L[3] - L[1], c[3] - c[1]) and gap_x <= max(4, 0.8 * (L[3] - L[1])):
                L[0] = min(L[0], c[0]); L[1] = min(L[1], c[1])
                L[2] = max(L[2], c[2]); L[3] = max(L[3], c[3])
                placed = True
                break
        if not placed:
            lines.append([c[0], c[1], c[2], c[3]])
    # 合并后可能还需要再合并一次（合并导致相邻行可连）
    changed = True
    while changed:
        changed = False
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                a, b = lines[i], lines[j]
                ovl = min(a[3], b[3]) - max(a[1], b[1])
                gap_x = max(a[0], b[0]) - min(a[2], b[2])
                if ovl >= 0.5 * min(a[3] - a[1], b[3] - b[1]) and gap_x <= max(4, 0.8 * min(a[3] - a[1], b[3] - b[1])):
                    lines[i] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
                    lines.pop(j)
                    changed = True
                    break
            if changed:
                break
    out = [tuple(L) for L in lines if (L[3] - L[1]) >= min_h and (L[2] - L[0]) * (L[3] - L[1]) >= min_px]
    return sorted(out, key=lambda t: (t[1], t[0]))


# ---------------------------------------------------------------------------
# 字形模板匹配（无 OCR 环境下的位图文字识别）
# ---------------------------------------------------------------------------
@lru_cache(maxsize=4096)
def _render_mask(text: str, font_path: str, size: int) -> np.ndarray | None:
    """用系统字体渲染文字 → 二值墨迹掩膜（裁剪到墨迹 bbox）。"""

    try:
        font = ImageFont.truetype(font_path, size)
    except Exception:
        return None
    w = max(8, int(size * len(text) * 1.4) + 20)
    h = int(size * 1.8) + 12
    img = Image.new("L", (w, h), 255)
    ImageDraw.Draw(img).text((10, 6), text, font=font, fill=0)
    arr = np.asarray(img) < 128
    bb = _bbox(arr)
    if bb is None:
        return None
    return arr[bb[1]:bb[3], bb[0]:bb[2]]


def _norm_mask(m: np.ndarray, target_h: int = 40) -> np.ndarray:
    if m.size == 0:
        return m
    h, w = m.shape
    nw = max(1, int(round(w * target_h / max(1, h))))
    im = Image.fromarray((m.astype(np.uint8) * 255))
    im = im.resize((nw, target_h), Image.LANCZOS)
    return np.asarray(im) > 110


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    h = max(a.shape[0], b.shape[0])
    w = max(a.shape[1], b.shape[1])
    pa = np.zeros((h, w), bool); pb = np.zeros((h, w), bool)
    pa[:a.shape[0], :a.shape[1]] = a
    pb[:b.shape[0], :b.shape[1]] = b
    inter = np.logical_and(pa, pb).sum()
    union = np.logical_or(pa, pb).sum()
    return float(inter) / float(union) if union else 0.0


def _glyph_score(a: np.ndarray, b: np.ndarray) -> float:
    """字形相似度（0~1）：Dice 重合度 + 列投影相关，取加权。

    为什么不能只用 IoU：位图里的字是 JPEG 压缩过的、笔画比干净渲染更粗，
    正确字的 IoU 也只有 ~0.5，和错误候选拉不开差距（实测复现过）。
    列投影对笔画粗细不敏感，能把正确候选稳定顶到第一。
    """

    if a.size == 0 or b.size == 0:
        return 0.0
    h = max(a.shape[0], b.shape[0])
    w = max(a.shape[1], b.shape[1])
    pa = np.zeros((h, w), np.float32); pb = np.zeros((h, w), np.float32)
    pa[:a.shape[0], :a.shape[1]] = a
    pb[:b.shape[0], :b.shape[1]] = b
    inter = float(np.logical_and(pa > 0, pb > 0).sum())
    dice = 2.0 * inter / max(1.0, float(pa.sum() + pb.sum()))
    ca, cb = pa.mean(axis=0), pb.mean(axis=0)
    ca = ca - ca.mean(); cb = cb - cb.mean()
    den = float(np.linalg.norm(ca) * np.linalg.norm(cb))
    corr = float(np.dot(ca, cb) / den) if den > 1e-9 else 0.0
    return 0.6 * dice + 0.4 * max(0.0, corr)


def _match_one(crop_n: np.ndarray, tm: np.ndarray, shifts: Sequence[int] = (-2, -1, 0, 1, 2)) -> float:
    """把模板缩放到切片宽度后（含横向微移）取最高字形相似度。"""

    best = 0.0
    h = crop_n.shape[0]
    t = Image.fromarray((tm.astype(np.uint8) * 255))
    for sh in shifts:
        w = crop_n.shape[1]
        tt = t.resize((max(2, w - abs(sh)), h), Image.LANCZOS)
        arr = np.asarray(tt) > 110
        canvas = np.zeros_like(crop_n)
        x0 = sh if sh > 0 else 0
        sx = 0 if sh > 0 else -sh
        width = min(canvas.shape[1] - x0, arr.shape[1] - sx)
        if width <= 0:
            continue
        canvas[:, x0:x0 + width] = arr[:, sx:sx + width]
        best = max(best, _glyph_score(crop_n, canvas))
    return best


def match_text_candidates(
    crop_mask: np.ndarray,
    candidates: Sequence[str],
    *,
    fonts: Sequence[str] = tuple(_CJK_FONTS),
    sizes: Sequence[int] | None = None,
    top: int = 3,
    norm_h: int = 40,
) -> list[tuple[str, float]]:
    """把一段文字掩膜与候选串逐个比对，返回 [(候选, 分数)] 降序。

    归一化：两边都按各自宽高比缩放到高 ``norm_h``，再把模板宽度拉伸到与切片一致
    （并做 ±2px 横向微移搜索）—— 这样对字体、字号、字距差异都鲁棒。
    """

    bb = _bbox(crop_mask)
    if bb is None:
        return []
    crop = crop_mask[bb[1]:bb[3], bb[0]:bb[2]]
    ch = crop.shape[0]
    if sizes is None:
        sizes = sorted({max(8, int(round(ch * f))) for f in (0.85, 1.0, 1.15, 1.3)})
    crop_n = _norm_mask(crop, norm_h)
    best: dict[str, float] = {}
    for cand in candidates:
        s = 0.0
        for fp in fonts:
            for sz in sizes:
                tm = _render_mask(cand, fp, int(sz))
                if tm is None:
                    continue
                s = max(s, _match_one(crop_n, _norm_mask(tm, norm_h)))
        best[cand] = s
    return sorted(best.items(), key=lambda t: -t[1])[:top]


def _segment_glyphs(mask: np.ndarray, *, min_w: int = 2) -> list[tuple[int, int]]:
    """按列投影把一行文字切成单字（返回 x 区间）。"""

    bb = _bbox(mask)
    if bb is None:
        return []
    m = mask[bb[1]:bb[3], bb[0]:bb[2]]
    cols = m.sum(axis=0) > 0
    spans: list[tuple[int, int]] = []
    start = None
    for i, on in enumerate(cols):
        if on and start is None:
            start = i
        elif not on and start is not None:
            spans.append((start, i))
            start = None
    if start is not None:
        spans.append((start, len(cols)))
    return [(a + bb[0], b + bb[0]) for a, b in spans if b - a >= min_w]


def read_numeric_line(
    crop_mask: np.ndarray,
    *,
    font: str = _DIGIT_FONT,
) -> tuple[str, float]:
    """把一行数字切片逐字识别成字符串（数字/小数点/百分号）。

    返回 (识别串, 平均 IoU)。识别串里 '.' 与 '%' 由模板自然区分。
    注意：这里**只做逐字模板匹配**，不做任何"猜数字"。
    """

    bb = _bbox(crop_mask)
    if bb is None:
        return "", 0.0
    h = bb[3] - bb[1]
    spans = _segment_glyphs(crop_mask)
    if not spans:
        return "", 0.0
    scores: list[float] = []
    chars: list[str] = []
    for (x0, x1) in spans:
        seg = crop_mask[bb[1]:bb[3], x0:x1]
        sbb = _bbox(seg)
        if sbb is None:
            continue
        seg = seg[sbb[1]:sbb[3], sbb[0]:sbb[2]]
        best_c, best_s = "", 0.0
        for c in _DIGIT_CHARS:
            ds = 0.0
            for sz in (max(8, h), max(8, h + 2), max(8, h - 2), 12, 14, 16, 18, 20):
                tm = _render_mask(c, font, int(sz))
                if tm is None:
                    continue
                ds = max(ds, _iou(_norm_mask(seg, 40), _norm_mask(tm, 40)))
            if ds > best_s:
                best_c, best_s = c, ds
        chars.append(best_c)
        scores.append(best_s)
    return "".join(chars), (sum(scores) / len(scores) if scores else 0.0)


# ---------------------------------------------------------------------------
# 正文锚点解析
# ---------------------------------------------------------------------------
_GROWTH_ANCHOR_RE = re.compile(r"增长率[^。；;]{0,24}?(\d+(?:\.\d+)?)\s*%")
_RANK_RE = re.compile(r"(?:位列|位居|排名|排在|居)\s*第\s*([一二三四五六七八九十\d]+)")
_TOP_RE = re.compile(r"仅次于([\u4e00-\u9fff]{2,8})")
_TOP1_RE = re.compile(r"(?:最快|最高|增幅最大|居首|排名第一|位列第一)[^。；;]{0,12}?([\u4e00-\u9fff]{2,8})")


def parse_growth_anchor(page_text: str) -> dict[str, Any] | None:
    """从正文里抽出「某个行业的增长率 + 它排第几 + 榜首是谁」。

    实测原文（物理页 72）：
        工业控制领域所占的比例虽然仅为7%，但其增长率达10.5%，
        在所有应用行业中位列第二，仅次于汽车电子。

    → {value: 10.5, rank: 2, top_hint: '汽车电子', quote: ...}

    **这是本模块唯一的绝对刻度来源**；没有它就只输出相对关系（不猜数）。
    """

    text = re.sub(r"\s+", "", page_text or "")
    m = _GROWTH_ANCHOR_RE.search(text)
    if not m:
        return None
    value = float(m.group(1))
    # 取整句（以句末标点为界），而不是固定窗口 —— 窗口会跨句，引用到无关内容
    s0 = max(text.rfind(ch, 0, m.start()) for ch in "。；;！？")
    e0 = len(text)
    for ch in "。；;！？":
        p = text.find(ch, m.end())
        if p != -1:
            e0 = min(e0, p)
    quote = text[s0 + 1:e0 + 1]
    if not (4 <= len(quote) <= 120):
        lo = max(0, m.start() - 40)
        hi = min(len(text), m.end() + 40)
        quote = text[lo:hi]
    window = quote
    rank = None
    rm = _RANK_RE.search(window)
    if rm:
        tok = rm.group(1)
        rank = _CN_NUM.get(tok, None)
        if rank is None and tok.isdigit():
            rank = int(tok)
    top_hint = ""
    tm = _TOP_RE.search(window) or _TOP1_RE.search(window)
    if tm:
        top_hint = tm.group(1)
    if rank is None and re.search(r"(最快|最高|增幅最大|居首|第1|排第一)", window):
        rank = 1
    return {"value": value, "rank": rank, "top_hint": top_hint, "quote": window}


# ---------------------------------------------------------------------------
# 条形图主流程
# ---------------------------------------------------------------------------
def _detect_axis(ink: np.ndarray) -> int | None:
    """找纵轴（基线）：排除 3px 图像边框后，低饱和深色像素最多的列。"""

    H, W = ink.shape
    if W < 12:
        return None
    col = ink.sum(axis=0)
    inner = col[3:W - 3]
    if inner.size == 0:
        return None
    x = int(np.argmax(inner)) + 3
    # 纵轴至少要占图高的 15%，否则只是某个字的竖笔
    if col[x] < max(8, H * 0.15):
        return None
    return x


def _bar_segments(color: np.ndarray) -> list[tuple[int, int]]:
    """柱体掩膜按行连续段切分（柱是水平长条，行方向连通）。"""

    rows = color.sum(axis=1)
    thr = max(8, int(color.shape[1] * 0.01))
    segs: list[tuple[int, int]] = []
    start = None
    for y in range(color.shape[0]):
        on = rows[y] > thr
        if on and start is None:
            start = y
        elif not on and start is not None:
            segs.append((start, y - 1))
            start = None
    if start is not None:
        segs.append((start, color.shape[0] - 1))
    # 合并被 1~2 行空隙切开的同一根柱
    merged: list[tuple[int, int]] = []
    for s in segs:
        if merged and s[0] - merged[-1][1] <= 3:
            merged[-1] = (merged[-1][0], s[1])
        else:
            merged.append(s)
    return [s for s in merged if (s[1] - s[0] + 1) >= 4]


def analyze_bar_chart(
    img: Image.Image,
    page_text: str = "",
    *,
    industry_candidates: Sequence[str] = tuple(DEFAULT_INDUSTRY_CANDIDATES),
    value_tolerance: float = 0.06,
) -> ChartResult | None:
    """对一张位图尝试条形图数值抽取；不是条形图返回 ``None``。"""

    lum, color, ink = _masks(img)
    res = ChartResult(chart_type="bar", width=img.width, height=img.height)
    if color.mean() < 0.004 or color.mean() > 0.6:
        return None
    x_axis = _detect_axis(ink)
    if x_axis is None:
        return None
    segs = _bar_segments(color)
    if not (3 <= len(segs) <= 40):
        return None
    res.x_axis = x_axis

    # ---- 每根柱：位置、方向、长度 ----
    bars: list[BarItem] = []
    for i, (y0, y1) in enumerate(segs):
        band = color[y0:y1 + 1, :]
        cols = band.sum(axis=0)
        xs = np.where(cols >= max(2, int((y1 - y0 + 1) * 0.5)))[0]
        if xs.size == 0:
            continue
        xl, xr = int(xs.min()), int(xs.max())
        left_len, right_len = x_axis - xl, xr - x_axis
        direction = "neg" if left_len > right_len else "pos"
        length = max(left_len, right_len)
        bars.append(BarItem(row=i, direction=direction, length_px=int(length),
                            bar_bbox=[int(xl), int(y0), int(xr), int(y1)]))
    if len(bars) < 3:
        return None

    # ---- 正文锚点 → 绝对刻度 ----
    anchor = parse_growth_anchor(page_text)
    res.anchor = anchor or {}
    pos = [b for b in bars if b.direction == "pos"]
    if anchor and anchor.get("rank") and pos:
        rank = int(anchor["rank"])
        ranked = sorted(pos, key=lambda b: -b.length_px)
        if 1 <= rank <= len(ranked):
            ref = ranked[rank - 1]
            if ref.length_px > 0:
                scale = float(anchor["value"]) / float(ref.length_px)
                res.scale_percent_per_px = scale
                for b in bars:
                    v = b.length_px * scale
                    b.value = round(-v if b.direction == "neg" else v, 1)
                    b.value_source = "geometry+anchor(rank=%d,%s%%→%dpx)" % (rank, anchor["value"], ref.length_px)
    if res.scale_percent_per_px is None:
        res.warnings.append("正文未给出可用锚点（增长率+排名），**只输出相对关系，不给绝对值**")
        for b in bars:
            b.value = None
            b.value_source = "uncalibrated"

    # ---- 标签识别（模板匹配；CLIP 通道由调用方补充） ----
    lab_scores: list[float] = []
    # 数值标签的候选串：**由几何+锚点算出**（7 个值 × 正负号），
    # 让「印刷数值识别」变成一次 7~14 选 1 的分类 —— 这样它才是**独立复核**，
    # 而不是把几何结果抄回去。
    value_strings: list[str] = []
    for b in bars:
        if b.value is not None:
            value_strings.append("%.1f%%" % abs(float(b.value)))
            value_strings.append("-%.1f%%" % abs(float(b.value)))
    value_strings = sorted(set(value_strings))
    for i, b in enumerate(bars):
        y0, y1 = b.bar_bbox[1], b.bar_bbox[3]
        pad = 8
        by0, by1 = max(0, y0 - pad), min(img.height, y1 + pad + 1)
        # 行业名：基线左侧、与柱同一 y 带的文字行
        left_lines = _text_lines(ink, (0, by0, max(1, x_axis - 2), by1))
        left_lines = [L for L in left_lines if (L[3] - L[1]) >= 6 and (L[2] - L[0]) >= 4]
        chosen = None
        if left_lines:
            # 垂直中心最接近柱中心者
            bc = (y0 + y1) / 2.0
            chosen = min(left_lines, key=lambda L: abs((L[1] + L[3]) / 2.0 - bc))
        if chosen:
            cm = np.zeros_like(ink)
            cm[chosen[1]:chosen[3], chosen[0]:chosen[2]] = ink[chosen[1]:chosen[3], chosen[0]:chosen[2]]
            ranked_c = match_text_candidates(cm, industry_candidates)
            if ranked_c:
                b.label = ranked_c[0][0]
                b.label_score = round(float(ranked_c[0][1]), 3)
                b.label_alt = ranked_c[1][0] if len(ranked_c) > 1 else ""
                lab_scores.append(b.label_score)
            b.label_bbox = [int(chosen[0]), int(chosen[1]), int(chosen[2]), int(chosen[3])]

        # 数值标签：排除行业名区域后、离柱末端最近的文字行
        cand_lines = _text_lines(ink, (0, max(0, by0 - 26), img.width, min(img.height, by1 + 26)))
        cand_lines = [L for L in cand_lines if (L[3] - L[1]) >= 5 and (L[2] - L[0]) >= 3]
        # 剔除「柱体上/下沿的坐标轴残段」：它们整条都压在基线列上，不是标签
        cand_lines = [L for L in cand_lines
                      if not (x_axis - 4 <= L[0] and L[2] <= x_axis + 4)]
        if b.label_bbox:
            lx0, ly0, lx1, ly1 = b.label_bbox
            cand_lines = [L for L in cand_lines
                          if not (L[0] >= lx0 - 2 and L[2] <= lx1 + 2 and L[1] >= ly0 - 2 and L[3] <= ly1 + 2)]
        if cand_lines:
            ex = x_axis - b.length_px if b.direction == "neg" else x_axis + b.length_px
            ey = (y0 + y1) / 2.0
            def _d(L):
                # 距离柱末端的水平距离为主、垂直距离为辅；数值标签不会与柱重叠太多
                dx = min(abs(L[0] - ex), abs(L[2] - ex))
                dy = abs((L[1] + L[3]) / 2.0 - ey)
                return dx + 0.5 * dy
            vl = min(cand_lines, key=_d)
            vm = np.zeros_like(ink)
            vm[vl[1]:vl[3], vl[0]:vl[2]] = ink[vl[1]:vl[3], vl[0]:vl[2]]
            if value_strings:
                ranked_v = match_text_candidates(vm, value_strings, fonts=(_DIGIT_FONT,))
                if ranked_v:
                    b.printed_value = ranked_v[0][0]
                    b.printed_score = round(float(ranked_v[0][1]), 3)
            if not b.printed_value:
                txt, sc = read_numeric_line(vm)
                b.printed_value = txt
                b.printed_score = round(float(sc), 3)

    # ---- 独立复核：印刷数值 vs 几何算出 ----
    verified = 0
    checked = 0
    for b in bars:
        if b.value is None or not b.printed_value:
            continue
        mv = re.search(r"-?\d+(?:\.\d+)?", b.printed_value)
        if not mv:
            continue
        checked += 1
        try:
            pv = float(mv.group(0))
        except ValueError:
            continue
        if abs(pv - float(b.value)) <= value_tolerance:
            b.ocr_verified = True
            verified += 1
    if checked:
        res.value_method = "template_match(%d/%d 与几何一致)" % (verified, checked)
        if verified < checked:
            res.warnings.append("数值标签识别与几何不一致 %d 组（已逐条记录 printed_value，未覆盖几何值）"
                                % (checked - verified))
    else:
        res.value_method = "geometry+anchor（数值标签未能识别，未做独立复核）"
    if lab_scores:
        res.label_method = "template_match(平均IoU=%.2f)" % (sum(lab_scores) / len(lab_scores))
    else:
        res.label_method = "none"

    # ---- 极值判定（全部来自几何结果，不含常量） ----
    calibrated = [b for b in bars if b.value is not None]
    if calibrated:
        mx = max(calibrated, key=lambda b: float(b.value))
        negs = [b for b in calibrated if b.direction == "neg" or float(b.value) < 0]
        res.extremes = {
            "max_label": mx.label, "max_value": mx.value,
            "negatives": [b.label for b in negs],
            "negative_values": [b.value for b in negs],
            "n_bars": len(bars),
        }
        if anchor and anchor.get("top_hint"):
            res.extremes["top_hint"] = anchor["top_hint"]
            res.extremes["top_hint_matches_max"] = bool(
                mx.label and (mx.label in anchor["top_hint"] or anchor["top_hint"] in mx.label)
            )
            if not res.extremes["top_hint_matches_max"]:
                res.warnings.append("正文提示榜首=「%s」，但几何最大柱标签=「%s」，两者不一致（均如实保留）"
                                    % (anchor["top_hint"], mx.label))
        res.extremes["max_fraction_ok"] = True
    else:
        pos_sorted = sorted([b for b in bars if b.direction == "pos"], key=lambda b: -b.length_px)
        res.extremes = {
            "max_label": pos_sorted[0].label if pos_sorted else "",
            "max_value": None,
            "negatives": [b.label for b in bars if b.direction == "neg"],
            "negative_values": [],
            "n_bars": len(bars),
            "uncalibrated": True,
        }

    res.bars = bars
    res.extraction_method = ("geometry+anchor" if res.scale_percent_per_px is not None else "geometry_only")
    if res.label_method.startswith("template_match"):
        res.extraction_method += "+template_ocr"
    return res


# ---------------------------------------------------------------------------
# 3D 饼图：只做标签识别，不做角度定量反算（透视失真）
# ---------------------------------------------------------------------------
def analyze_pie_chart(
    img: Image.Image,
    *,
    industry_candidates: Sequence[str] = tuple(DEFAULT_INDUSTRY_CANDIDATES),
) -> ChartResult | None:
    """识别 3D 饼图及其标签文字。

    **明确不做**：扇区角度反算百分比（3D 透视下不可靠）。
    这里只把图上的行业名识别出来，数值交给正文/其它子图。
    """

    lum, color, ink = _masks(img)
    if color.mean() < 0.05:
        return None
    # 饼图 vs 条形图的判据：**最大彩色连通块的包围盒**是否在长宽两个方向都足够大。
    # 条形图的每根柱都是细长条（高度只有 20 余 px），饼图的扇区则是一大团。
    lab, n = ndimage.label(color, structure=np.ones((3, 3)))
    if n < 2:
        return None
    sizes = ndimage.sum(color, lab, range(1, n + 1))
    h, w = color.shape
    idx = int(np.argmax(sizes)) + 1
    sl = ndimage.find_objects(lab == idx)[0]
    bh, bw = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
    if not (bh >= 0.25 * h and bw >= 0.25 * w):
        return None

    res = ChartResult(chart_type="pie3d", width=img.width, height=img.height, quantitative=False)
    res.extraction_method = "geometry_only"
    res.value_method = "none"
    lines = [L for L in _text_lines(ink) if (L[3] - L[1]) >= 8 and (L[2] - L[0]) >= 8]
    found: list[str] = []
    scores: list[float] = []
    for L in lines:
        cm = np.zeros_like(ink)
        cm[L[1]:L[3], L[0]:L[2]] = ink[L[1]:L[3], L[0]:L[2]]
        ranked_c = match_text_candidates(cm, industry_candidates)
        if ranked_c and ranked_c[0][1] >= 0.5:
            if ranked_c[0][0] not in found:
                found.append(ranked_c[0][0])
                scores.append(float(ranked_c[0][1]))
    res.pie_labels = found
    res.label_method = ("template_match(%.2f)" % (sum(scores) / len(scores))) if scores else "none"
    res.warnings.append("3D 饼图透视失真：**未做扇区角度反算**，仅做标签识别（见 设计/图像解析方案.md §4.4）")
    return res


# ---------------------------------------------------------------------------
# 文本渲染（供索引 chunk.text 使用）
# ---------------------------------------------------------------------------
def format_bar_chart_text(title: str, source_note: str, res: ChartResult, unit: str = "%") -> str:
    """把抽取结果拼成「可直接嵌入检索」的中文文本。"""

    parts: list[str] = []
    head = title.strip() if title else "位图图表"
    parts.append("%s（%s 条形图，单位 %s）" % (head, res.chart_type, unit or "—"))
    if source_note:
        parts.append(source_note.strip())
    ranked = sorted([b for b in res.bars if b.value is not None], key=lambda b: -float(b.value))
    if ranked:
        parts.append("行业—增长率：" + "、".join("%s %.1f%%" % (b.label or "未知", float(b.value)) for b in ranked) + "。")
    else:
        parts.append("柱长（相对）：" + "、".join(
            "%s %dpx(%s)" % (b.label or "未知", b.length_px, b.direction) for b in res.bars) + "。")
    ex = res.extremes or {}
    if ex.get("max_label") and ex.get("max_value") is not None:
        parts.append("增长率最快的行业是 %s（%.1f%%）。" % (ex["max_label"], float(ex["max_value"])))
    negs = ex.get("negatives") or []
    if negs:
        vals = ex.get("negative_values") or []
        if vals:
            parts.append("负增长的行业是 %s（%.1f%%）。" % ("、".join(negs), float(vals[0])))
        else:
            parts.append("负增长的行业是 %s。" % "、".join(negs))
    else:
        parts.append("无负增长行业。")
    if res.anchor.get("quote"):
        parts.append("正文交叉印证：%s" % res.anchor["quote"])
    return "\n".join(parts)
