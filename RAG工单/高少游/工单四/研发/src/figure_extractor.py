# -*- coding: utf-8 -*-
"""图形区域抽取模块（PDF 图像内容解析第一步）
工单编号: 人工智能 NLP-RAG-图像内容解析及检索优化

本模块负责从 PDF 中「定位并抽取图形区域」，是图像语义解析的前置步骤。
招股说明书中的图形有两类载体，必须分别处理：

1. **嵌入位图（raster）**：如「2008 年中国 IC 市场应用结构与增长图」，以 JPEG/PNG
   位图嵌入页面。特点：图内文字（坐标轴标签、数值）不可被文本层抽取。
2. **矢量绘图簇（vector）**：如「公司组织结构图」，由大量矩形（节点框）与线段
   （连线）绘制而成。特点：图内文字在文本层中可抽取，但阅读顺序被打散
   （常为竖排单字），必须按「节点框 + 连线」还原结构。

关键技术点：
- **水印剔除**：招股说明书每页平铺 15 个相同水印位图，用「xref 跨页出现频率」
  识别并剔除，避免把水印误当图形；
- **页眉页脚横线剔除**：页面顶部贯穿全宽的发丝线是版式线而非图形；
- **表格区域排除**：表格边框同样是大量线段/矩形，用表格 bbox 做重叠排除，
  避免把表格误判为图形；
- **矢量簇聚类**：对绘图元素做并查集聚类（带间隙膨胀），得到完整图形区域；
- **区域渲染**：把图形区域按倍率渲染为 PNG，作为多模态模型（CLIP/VLM）的输入。

产出 `FigureRegion` 列表，供 `figure_semantics`（语义还原）与 `image_index`
（CLIP 跨模态索引）使用。
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from src import config

logger = logging.getLogger(__name__)

try:
    import pymupdf
except ImportError:      # 兼容旧版本包名
    import fitz as pymupdf

BBox = Tuple[float, float, float, float]

# 图形标题线索：用于从邻近文本中识别「图题」
_FIG_CAP_RE = re.compile(
    r"(图\s*\d|结构图|流程图|示意图|趋势图|增长图|饼图|柱状图|折线图|框架图|如下图|图所示)"
)
# 页眉页脚特征词（版式文字，不可作为图题）
_BOILER_RE = re.compile(r"(招股意向书|招股说明书|首次公开发行|页码)")


@dataclass
class FigureRegion:
    """一个被抽取出的图形区域。"""

    source: str                                   # 所属文档文件名
    page: int                                     # 页码（1 基）
    bbox: BBox = (0.0, 0.0, 0.0, 0.0)             # 区域边界（PDF 点）
    kind: str = "vector"                          # raster / vector
    image_path: str = ""                          # 渲染后的 PNG 路径
    caption: str = ""                             # 图题
    texts: List[dict] = field(default_factory=list)   # 区域内文字（含坐标）
    n_boxes: int = 0                              # 区域内矩形数（节点框）
    n_lines: int = 0                              # 区域内线段数（连线）
    figure_type: str = ""                         # 多模态模型判定的图形类型
    type_score: float = 0.0                       # 类型判定置信度

    @property
    def key(self) -> str:
        return f"{self.source}#p{self.page}#{int(self.bbox[0])}_{int(self.bbox[1])}"

    @property
    def page_label(self) -> str:
        return f"《{self.source}》第{self.page}页"


# ---------------- 基础工具 -----------------------------------------------------
def _norm(text: str) -> str:
    return re.sub(r"[ \t\u3000]+", " ", (text or "").replace("\n", " ")).strip()


def _area(bbox: BBox) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def _overlap_ratio(a: BBox, b: BBox) -> float:
    """a 与 b 的交集面积占 a 面积的比例。"""
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    return inter / _area(a) if _area(a) > 0 else 0.0


def _watermark_xrefs(doc, ratio: float = 0.30) -> set:
    """识别水印位图：在超过 ratio 比例的页面上重复出现的同一 xref。"""
    from collections import defaultdict

    freq: Dict[int, set] = defaultdict(set)
    n_pages = doc.page_count
    for i in range(n_pages):
        try:
            for img in doc.load_page(i).get_images(full=True):
                freq[img[0]].add(i)
        except Exception:
            continue
    threshold = max(3, int(n_pages * ratio))
    return {x for x, pages in freq.items() if len(pages) >= threshold}


def _is_rule(rect, page_rect) -> bool:
    """是否为版式发丝线（贯穿全宽的页眉/页脚横线）。"""
    h = rect.y1 - rect.y0
    w = rect.x1 - rect.x0
    if h <= 3 and w >= 0.7 * (page_rect.x1 - page_rect.x0):
        return True
    if w <= 3 and h >= 0.7 * (page_rect.y1 - page_rect.y0):
        return True
    return False


def _cluster(rects: List, gap: float = 40.0) -> List[Tuple[BBox, List[int]]]:
    """并查集聚类：把间隙小于 gap 的矩形/线段合并为同一图形区域。"""
    n = len(rects)
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    boxes = [(r.x0, r.y0, r.x1, r.y1) for r in rects]
    # 网格分桶加速：避免 O(n^2) 比较
    cell = max(gap * 2, 40.0)
    buckets: Dict[Tuple[int, int], List[int]] = {}
    for i, b in enumerate(boxes):
        for cx in range(int(b[0] // cell), int(b[2] // cell) + 1):
            for cy in range(int(b[1] // cell), int(b[3] // cell) + 1):
                buckets.setdefault((cx, cy), []).append(i)
    for i, b in enumerate(boxes):
        ex = (b[0] - gap, b[1] - gap, b[2] + gap, b[3] + gap)
        cand = set()
        for cx in range(int(ex[0] // cell), int(ex[2] // cell) + 1):
            for cy in range(int(ex[1] // cell), int(ex[3] // cell) + 1):
                cand.update(buckets.get((cx, cy), []))
        for j in cand:
            if j <= i:
                continue
            c = boxes[j]
            if not (ex[2] < c[0] or ex[0] > c[2] or ex[3] < c[1] or ex[1] > c[3]):
                union(i, j)

    groups: Dict[int, List[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    out: List[Tuple[BBox, List[int]]] = []
    for idxs in groups.values():
        xs0 = min(boxes[i][0] for i in idxs)
        ys0 = min(boxes[i][1] for i in idxs)
        xs1 = max(boxes[i][2] for i in idxs)
        ys1 = max(boxes[i][3] for i in idxs)
        out.append(((xs0, ys0, xs1, ys1), idxs))
    return out


def _expand_to_rects(box: BBox, rects: List, margin: float = 2.0) -> BBox:
    """把簇边界扩展到覆盖「中心落在簇内」的所有矩形。

    矢量图形按「间隙聚类」时，图形边界上的节点可能与主体相距略大于聚类间隙而
    落入独立小簇（如组织结构图顶端的“股东大会”仅与下方节点纵向相连）。若不做
    扩展，区域上沿会被裁掉，该节点在语义还原时被丢弃，导致层级树根节点错误。
    """
    x0, y0, x1, y1 = box
    for r in rects:
        cx, cy = (r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2
        if x0 - margin <= cx <= x1 + margin and y0 - margin <= cy <= y1 + margin:
            x0, y0 = min(x0, r.x0), min(y0, r.y0)
            x1, y1 = max(x1, r.x1), max(y1, r.y1)
    return (x0, y0, x1, y1)


def _texts_in(page, bbox: BBox) -> List[dict]:
    """抽取落在区域内的文字 span（含坐标，供结构还原）。"""
    out: List[dict] = []
    try:
        d = page.get_text("dict")
    except Exception:
        return out
    x0, y0, x1, y1 = bbox
    for blk in d.get("blocks", []):
        if blk.get("type") != 0:
            continue
        for ln in blk.get("lines", []):
            for sp in ln.get("spans", []):
                t = _norm(sp.get("text", ""))
                if not t:
                    continue
                sx0, sy0, sx1, sy1 = sp["bbox"]
                cx, cy = (sx0 + sx1) / 2, (sy0 + sy1) / 2
                if x0 - 2 <= cx <= x1 + 2 and y0 - 2 <= cy <= y1 + 2:
                    out.append({"text": t, "x0": sx0, "y0": sy0, "x1": sx1, "y1": sy1})
    out.sort(key=lambda s: (round(s["y0"] / 6), s["x0"]))
    return out


# 资料来源类注释（位于图下方，不是图题）
_SOURCE_NOTE_RE = re.compile(r"(资料来源|数据来源|来源：|注：|单位：)")


def _caption(page, bbox: BBox, page_rect, max_gap: float = 40.0) -> str:
    """在图区上/下方寻找图题。

    打分规则：图题通常位于图形**上方**且为短句标题；「资料来源/数据来源」等注释
    虽更靠近图形，但属于出处说明而非图题，需降权。
    """
    x0, y0, x1, y1 = bbox
    best, best_score = "", -1e9
    try:
        blocks = page.get_text("blocks") or []
    except Exception:
        return ""
    for b in blocks:
        bx0, by0, bx1, by1 = b[0], b[1], b[2], b[3]
        text = _norm(b[4])
        if not text or len(text) > 80:
            continue
        if by1 < 62 or by0 > page_rect.y1 - 62:      # 页眉 / 页脚
            continue
        if _BOILER_RE.search(text):
            continue
        if bx1 < x0 - 12 or bx0 > x1 + 12:
            continue
        above, below = y0 - by1, by0 - y1
        if -2 <= above <= max_gap:
            gap, is_above = above, True
        elif -2 <= below <= max_gap:
            gap, is_above = below, False
        else:
            continue
        score = (max_gap - gap) / max_gap
        score += 1.2 if _FIG_CAP_RE.search(text) else 0.0
        score += 0.8 if is_above else 0.0                    # 图题多在图上方
        score -= 1.0 if _SOURCE_NOTE_RE.search(text) else 0.0  # 出处注释非图题
        if score > best_score:
            best, best_score = text, score
    return best


# ---------------- 主抽取逻辑 ---------------------------------------------------
def _is_noise_region(box: BBox, page_rect) -> bool:
    """版式噪声区域过滤。

    - 扫描签字页 / 整页扫描图：区域几乎覆盖整页，不是「插图」；
    - 横向条带：宽度接近整页、高度很小，是扫描页被切分的碎片或版式装饰。
    """
    pw = page_rect.x1 - page_rect.x0
    ph = page_rect.y1 - page_rect.y0
    w, h = box[2] - box[0], box[3] - box[1]
    if _area(box) / max(1.0, pw * ph) > 0.85:
        return True
    if w >= 0.95 * pw and h <= 0.35 * ph:
        return True
    return False


def _detect_page_figures(page, doc, page_rect, source: str,
                         watermark: set, table_boxes: List[BBox],
                         page_area: float) -> List[FigureRegion]:
    """检测单页中的图形区域（位图 + 矢量簇）。"""
    regions: List[FigureRegion] = []
    raw_boxes: List[Tuple[BBox, str]] = []       # (bbox, kind)

    # ---- A) 嵌入位图 ----
    seen_xref = set()
    for img in page.get_images(full=True):
        xref = img[0]
        if xref in watermark or xref in seen_xref:
            continue
        try:
            info = doc.extract_image(xref)
        except Exception:
            continue
        if min(info.get("width", 0), info.get("height", 0)) < config.FIGURE_MIN_PIXELS:
            continue
        for r in page.get_image_rects(xref):
            box = (r.x0, r.y0, r.x1, r.y1)
            if _area(box) / page_area < config.FIGURE_MIN_AREA_RATIO:
                continue
            if box[2] - box[0] < config.FIGURE_MIN_PT_WIDTH:
                continue
            if box[3] - box[1] < config.FIGURE_MIN_PT_HEIGHT:
                continue
            if _is_noise_region(box, page_rect):
                continue
            seen_xref.add(xref)
            raw_boxes.append((box, "raster"))

    # ---- B) 矢量绘图簇（组织结构图 / 流程图） ----
    try:
        drawings = page.get_drawings()
    except Exception:
        drawings = []
    rect_items, line_items = [], []
    for d in drawings:
        r = d.get("rect")
        if r is None or _is_rule(r, page_rect):
            continue
        kinds = {it[0] for it in d.get("items", [])}
        if kinds <= {"re"}:
            rect_items.append(r)
        else:
            line_items.append(r)

    for box, idxs in _cluster(rect_items, gap=40.0):
        n_rect = len(idxs)
        # 节点框过少 → 版式装饰；过多 → 大概率是「未识别成表格」的网格（表格边框）
        if not (4 <= n_rect <= config.FIGURE_MAX_BOXES):
            continue
        # 紧致化：把中心落在簇内的矩形一并纳入，避免图形边界节点（如组织结构图
        # 顶端的“股东大会”）被裁掉，从而保证后续层级还原的根节点正确。
        box = _expand_to_rects(box, rect_items)
        if _area(box) / page_area < config.FIGURE_MIN_AREA_RATIO:
            continue
        if _is_noise_region(box, page_rect):
            continue
        # 表格排除：与任一表格 bbox 重叠度 > 0.5 则判定为表格边框
        if any(_overlap_ratio(box, tb) > 0.5 for tb in table_boxes):
            continue
        raw_boxes.append((box, "vector"))

    # ---- C) 合并重叠区域（位图与矢量可能同属一张图） ----
    merged: List[Tuple[BBox, str]] = []
    for box, kind in raw_boxes:
        for i, (m, mk) in enumerate(merged):
            if _overlap_ratio(box, m) > 0.35 or _overlap_ratio(m, box) > 0.35:
                nx0, ny0 = min(box[0], m[0]), min(box[1], m[1])
                nx1, ny1 = max(box[2], m[2]), max(box[3], m[3])
                merged[i] = ((nx0, ny0, nx1, ny1), mk if mk == kind else f"{mk}+{kind}")
                break
        else:
            merged.append((box, kind))

    # ---- D) 组装区域对象（文字量校验） ----
    for box, kind in merged:
        texts = _texts_in(page, box)
        if kind.startswith("vector"):
            if not (config.FIGURE_MIN_TEXTS <= len(texts) <= config.FIGURE_MAX_TEXTS):
                continue
        elif not texts and _area(box) / page_area < config.FIGURE_MIN_AREA_RATIO:
            continue
        # 统计区域内矩形/线段数（供语义还原使用）
        n_boxes = sum(1 for r in rect_items
                      if box[0] - 2 <= (r.x0 + r.x1) / 2 <= box[2] + 2
                      and box[1] - 2 <= (r.y0 + r.y1) / 2 <= box[3] + 2)
        n_lines = sum(1 for r in line_items
                      if box[0] - 2 <= (r.x0 + r.x1) / 2 <= box[2] + 2
                      and box[1] - 2 <= (r.y0 + r.y1) / 2 <= box[3] + 2)
        regions.append(FigureRegion(
            source=source, page=page.number + 1, bbox=box, kind=kind,
            caption=_caption(page, box, page_rect), texts=texts,
            n_boxes=n_boxes, n_lines=n_lines,
        ))
    return regions


def extract_figures_from_pdf(pdf_path, out_dir=None,
                             table_boxes: Optional[Dict[int, List[BBox]]] = None
                             ) -> List[FigureRegion]:
    """抽取单个 PDF 的全部图形区域，并把区域渲染为 PNG。"""
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF 文件不存在: {path}")
    out_dir = Path(out_dir or config.FIGURE_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    regions: List[FigureRegion] = []
    with pymupdf.open(str(path)) as doc:
        watermark = _watermark_xrefs(doc)
        logger.info("[%s] 识别水印位图 %d 个", path.name, len(watermark))
        pending_caption = ""
        for i in range(doc.page_count):
            page = doc.load_page(i)
            page_rect = page.rect
            page_area = max(1.0, (page_rect.x1 - page_rect.x0) * (page_rect.y1 - page_rect.y0))
            tb = (table_boxes or {}).get(i + 1, [])
            figs = _detect_page_figures(page, doc, page_rect, path.name,
                                        watermark, tb, page_area)
            # 跨页图题继承：上一页末尾若出现“如下图”类图题，则本页图形沿用
            for fig in figs:
                if not fig.caption and pending_caption:
                    fig.caption = pending_caption
                fig.image_path = str(_render_region(page, fig, out_dir))
                regions.append(fig)
            pending_caption = _pending_caption(page, page_rect) or pending_caption
    logger.info("图形抽取：%s → %d 个图形区域", path.name, len(regions))
    return regions


def _pending_caption(page, page_rect) -> str:
    """取本页末尾的图题（供下一页图形继承）。"""
    try:
        blocks = page.get_text("blocks") or []
    except Exception:
        return ""
    for b in reversed(blocks):
        text = _norm(b[4])
        if not text or len(text) > 80:
            continue
        if b[1] < 62 or b[3] > page_rect.y1 - 62 or _BOILER_RE.search(text):
            continue
        if _FIG_CAP_RE.search(text):
            return text
    return ""


def _render_region(page, fig: FigureRegion, out_dir: Path) -> Path:
    """把图形区域按倍率渲染为 PNG（多模态模型输入）。"""
    stem = Path(fig.source).stem
    fn = out_dir / f"{stem}_p{fig.page}_{int(fig.bbox[0])}_{int(fig.bbox[1])}.png"
    try:
        zoom = config.FIGURE_RENDER_ZOOM
        clip = pymupdf.Rect(*fig.bbox)
        pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), clip=clip, alpha=False)
        pix.save(str(fn))
    except Exception as exc:
        logger.warning("图形渲染失败 %s: %s", fig.key, exc)
        return Path("")
    return fn


def extract_all_figures(pdf_paths: Sequence = None, out_dir=None) -> List[FigureRegion]:
    """抽取全部知识库文档的图形区域（含表格 bbox 排除）。"""
    paths = list(pdf_paths or config.PDF_PATHS)
    table_boxes: Dict[str, Dict[int, List[BBox]]] = {}
    if config.USE_TABLE_PARSER:
        try:
            from src.table_parser import parse_tables

            for p in paths:
                tb: Dict[int, List[BBox]] = {}
                for blk in parse_tables(p):
                    if blk.bbox:
                        tb.setdefault(blk.page, []).append(tuple(blk.bbox))
                table_boxes[str(p)] = tb
        except Exception as exc:
            logger.warning("表格 bbox 预取失败（图形检测将不做表格排除）: %s", exc)

    out: List[FigureRegion] = []
    for p in paths:
        out.extend(extract_figures_from_pdf(p, out_dir, table_boxes.get(str(p))))
    return out


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    logging.basicConfig(level=logging.INFO)
    figs = extract_all_figures()
    print(f"共抽取图形区域 {len(figs)} 个")
    for f in figs:
        print(f"  {f.key} kind={f.kind} boxes={f.n_boxes} lines={f.n_lines} "
              f"texts={len(f.texts)} caption={f.caption[:40]!r}")