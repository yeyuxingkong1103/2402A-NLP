# -*- coding: utf-8 -*-
# 【图像内容解析模块 · image_extractor.py】嵌入图片/矢量图抽取、图注邻近正文关联、竖排文本重建、离线OCR
# 工单编号：人工智能NLP-RAG-图像内容解析及检索优化

"""图像内容解析层（工单四核心新增）：

1. 嵌入位图：PyMuPDF枚举XObject，按“跨页复用次数+尺寸+长宽比+面积”过滤
   页眉Logo/水印与小图标，保存原图并用Pillow生成缩略图；
2. 矢量图形：图注门控（组织结构图/流程图/示意图等）+绘图密度检测，
   按绘图簇bbox把页面区域渲染成PNG，覆盖无嵌入位图的结构图/流程图；
3. 关联：为每张图绑定页码、图题、“如下图”引导句（支持跨页）、资料来源、
   纵向±130pt内邻近正文；
4. 图内文字：矢量横排文本 + 竖排单字按x聚类重建（保留bbox）；
   位图/渲染图走PaddleOCR离线OCR（PP-OCRv6本地缓存，禁止联网下载）；
5. CLIP/多模态大模型保留统一可插拔接口；本机无缓存时输出“伪多模态”图像文本块。
"""
import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pymupdf

from config import CONFIG

logger = logging.getLogger(__name__)

# 竖排单字判定：单个中日韩字符（含括号/顿号等图内常用符号）
_CJK_CHAR = re.compile(r"^[\u4e00-\u9fa5（）()、·\w]$")
_PCT_TOKEN = re.compile(r"^-?\d+(?:\.\d+)?%$")
_SOURCE_RE = re.compile(r"资料来源|数据来源")


@dataclass
class Label:
    """图内文字标签（横排或竖排重建结果，带PDF点坐标）。"""

    text: str
    bbox: Tuple[float, float, float, float]

    @property
    def cx(self) -> float:
        """标签中心x。"""
        return (self.bbox[0] + self.bbox[2]) / 2

    @property
    def cy(self) -> float:
        """标签中心y。"""
        return (self.bbox[1] + self.bbox[3]) / 2


@dataclass
class OcrItem:
    """OCR识别条目（坐标归一化到0~1，便于跨图片复用解题规则）。"""

    text: str
    score: float
    x: float
    y: float


@dataclass
class FigureRecord:
    """一张图的完整证据记录：图像文件 + 图注/邻近正文 + 图内文字。"""

    fig_id: str
    doc: str
    page_no: int
    kind: str                       # raster=嵌入位图 / vector=矢量渲染图
    bbox: Tuple[float, float, float, float]
    image_path: str = ""
    thumb_path: str = ""
    xref: int = 0
    caption: str = ""               # 图题
    ref_sentence: str = ""          # “……如下图”引导句
    source_line: str = ""           # 资料/数据来源
    nearby_text: str = ""           # 邻近正文
    labels: List[Label] = field(default_factory=list)   # 矢量文本标签
    ocr_items: List[OcrItem] = field(default_factory=list)
    h_lines: List[Tuple[float, float, float]] = field(default_factory=list)
    degraded: bool = False          # OCR不可用、仅靠图注邻近正文时为True

    def ocr_text(self) -> str:
        """拼接OCR文本（按从上到下、从左到右排序）。"""
        items = sorted(self.ocr_items, key=lambda it: (round(it.y / 0.05), it.x))
        return " ".join(it.text for it in items)

    def inner_text(self) -> str:
        """拼接图内全部文字：矢量标签 + OCR文本，去重保序。"""
        parts: List[str] = []
        for lb in self.labels:
            if lb.text and lb.text not in parts:
                parts.append(lb.text)
        for it in self.ocr_items:
            if it.text and it.text not in parts:
                parts.append(it.text)
        return "；".join(parts)

    def to_text(self) -> str:
        """生成可检索的“图像证据文本块”（伪多模态降级的核心载体）。"""
        kind_cn = "嵌入图片" if self.kind == "raster" else "矢量图形渲染图"
        lines = [f"【图像证据｜{self.doc}｜第{self.page_no}页｜{kind_cn}】"]
        if self.caption:
            lines.append(f"图题：{self.caption}")
        inner = self.inner_text()
        if inner:
            lines.append(f"图内文字：{inner}")
        if self.ref_sentence:
            lines.append(f"图注引导：{self.ref_sentence}")
        if self.nearby_text:
            lines.append(f"邻近正文：{self.nearby_text}")
        if self.source_line:
            lines.append(self.source_line)
        return "\n".join(lines)


class OcrEngine:
    """离线OCR引擎封装：优先本机已缓存模型的PaddleOCR（PP-OCRv6）。

    模型缓存在 ~/.paddlex/official_models，初始化与推理全程离线；
    Paddle 3.4的OneDNN/PIR存在兼容缺陷，启动前关闭mkldnn。
    任何初始化失败都安全降级（返回空结果，由调用方标注degraded）。
    """

    def __init__(self) -> None:
        """检测并尝试加载OCR模型，加载结果记录在available标志。"""
        self.available = False
        self.engine = None
        if not CONFIG.enable_ocr:
            logger.info("配置已关闭OCR，图像走图注+邻近正文降级")
            return
        try:
            os.environ.setdefault("FLAGS_use_mkldnn", "0")
            from paddleocr import PaddleOCR  # 延迟导入，避免无该依赖时影响启动
            self.engine = PaddleOCR(
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                enable_mkldnn=False, lang="ch")
            self.available = True
            logger.info("PaddleOCR离线引擎就绪（PP-OCRv6本地缓存）")
        except Exception as exc:
            logger.warning("PaddleOCR不可用，图像OCR降级为空：%s", exc)

    def recognize(self, image_path: str,
                  cache: Optional[dict] = None) -> List[OcrItem]:
        """对单张图片做OCR，结果按内容哈希缓存（建库可断点续跑）。

        :param image_path: 图片绝对路径
        :param cache: OCR结果缓存字典（持久化到image_store）
        :return: OcrItem列表（坐标按图片宽高归一化）
        """
        if not self.available:
            return []
        with open(image_path, "rb") as f:
            digest = hashlib.md5(f.read()).hexdigest()
        if cache is not None and digest in cache:
            return [OcrItem(**it) for it in cache[digest]]
        items: List[OcrItem] = []
        try:
            results = self.engine.predict(image_path)
            for result in results:
                texts = result.get("rec_texts", [])
                scores = result.get("rec_scores", [])
                polys = result.get("rec_polys", [])
                for idx, text in enumerate(texts):
                    text = (text or "").strip()
                    if not text:
                        continue
                    poly = polys[idx] if idx < len(polys) else None
                    if poly is not None:
                        xs = [float(p[0]) for p in poly]
                        ys = [float(p[1]) for p in poly]
                        w = max(xs) - min(xs) or 1.0
                        h = max(ys) - min(ys) or 1.0
                        x, y = min(xs) / w, min(ys) / h
                        # 归一化需要图片真实宽高，predict结果按原图像素，下面用宽高校正
                        items.append(OcrItem(text, round(float(scores[idx]), 4),
                                             float(min(xs)), float(min(ys))))
                    else:
                        items.append(OcrItem(text, round(float(scores[idx]), 4),
                                             0.0, 0.0))
            # 用原图宽高做真正归一化
            from PIL import Image
            with Image.open(image_path) as im:
                iw, ih = im.size
            items = [OcrItem(it.text, it.score,
                             round(min(max(it.x / iw, 0.0), 1.0), 4),
                             round(min(max(it.y / ih, 0.0), 1.0), 4))
                     for it in items]
        except Exception as exc:
            logger.warning("图片OCR失败 %s：%s", os.path.basename(image_path), exc)
            items = []
        if cache is not None:
            cache[digest] = [it.__dict__ for it in items]
        return items


class ImageEncoder:
    """CLIP/多模态大模型图像编码器统一接口（可插拔，本机离线时不启用）。

    在线实现约定：encode_images返回与文本BGE/CLIP同空间的L2归一化向量，
    使图像可以直接被跨模态稠密召回。检测到本地缓存权重时由multimodal_index
    自动实例化；本环境~/.cache/huggingface无CLIP且禁止联网，故保持关闭。
    """

    def __init__(self) -> None:
        """接口默认不可用；在线环境可用transformers/open_clip扩展实现。"""
        self.available = False
        self.dim = 0

    def encode_images(self, image_paths: List[str]):  # pragma: no cover
        """编码图像为向量（在线环境实现）。"""
        raise NotImplementedError

    def encode_texts(self, texts: List[str]):  # pragma: no cover
        """编码查询文本到同一向量空间（在线环境实现）。"""
        raise NotImplementedError


def _save_thumb(src_path: str) -> str:
    """用Pillow生成长边不超过配置值的缩略图（同目录加_thumb后缀）。

    :param src_path: 原图路径
    :return: 缩略图路径
    """
    from PIL import Image
    base, ext = os.path.splitext(src_path)
    thumb_path = base + "_thumb.jpg"
    with Image.open(src_path) as im:
        im = im.convert("RGB")
        im.thumbnail((CONFIG.thumb_max_side, CONFIG.thumb_max_side))
        im.save(thumb_path, "JPEG", quality=85)
    return thumb_path


def _union_bbox(rects: List[Tuple[float, float, float, float]],
                gap: float = 30.0) -> Optional[Tuple[float, float, float, float]]:
    """对矩形做基于间距的连通聚类，返回最大簇的并集bbox。

    :param rects: 矩形列表
    :param gap: 间距小于该值视为同一簇
    :return: 最大簇并集bbox，无矩形返回None
    """
    if not rects:
        return None
    clusters: List[List[Tuple[float, float, float, float]]] = []
    for rect in sorted(rects, key=lambda r: (r[1], r[0])):
        x0, y0, x1, y1 = rect
        placed = False
        for cluster in clusters:
            cx0 = min(r[0] for r in cluster) - gap
            cy0 = min(r[1] for r in cluster) - gap
            cx1 = max(r[2] for r in cluster) + gap
            cy1 = max(r[3] for r in cluster) + gap
            if x0 <= cx1 and x1 >= cx0 and y0 <= cy1 and y1 >= cy0:
                cluster.append(rect)
                placed = True
                break
        if not placed:
            clusters.append([rect])
    biggest = max(clusters, key=lambda c: (
        (max(r[2] for r in c) - min(r[0] for r in c))
        * (max(r[3] for r in c) - min(r[1] for r in c)), len(c)))
    return (min(r[0] for r in biggest), min(r[1] for r in biggest),
            max(r[2] for r in biggest), max(r[3] for r in biggest))


def _drawing_geometry(drawings: List[dict]
                      ) -> Tuple[List[Tuple[float, float, float, float]],
                                 List[Tuple[float, float, float]]]:
    """从PyMuPDF绘图指令中提取填充/描边矩形与水平线段。

    :param drawings: page.get_drawings()结果
    :return: (矩形列表, 水平线段[(y,x0,x1)])
    """
    rects: List[Tuple[float, float, float, float]] = []
    hlines: List[Tuple[float, float, float]] = []
    for draw in drawings:
        rect = draw.get("rect")
        if rect and rect.width > 2 and rect.height > 2:
            rects.append((rect.x0, rect.y0, rect.x1, rect.y1))
        for item in draw.get("items", []):
            if item[0] == "l":  # 直线段
                p1, p2 = item[1], item[2]
                if abs(p1.y - p2.y) < 1.0 and abs(p1.x - p2.x) > 8:
                    hlines.append((float((p1.y + p2.y) / 2),
                                   float(min(p1.x, p2.x)), float(max(p1.x, p2.x))))
            elif item[0] == "re":  # 矩形路径
                r = item[1]
                if r.width > 2 and r.height > 2:
                    rects.append((r.x0, r.y0, r.x1, r.y1))
    return rects, hlines


def _inner_labels(page, bbox: Tuple[float, float, float, float]
                  ) -> Tuple[List[Label], List[Tuple[float, float, float]]]:
    """提取图形bbox内的文本标签，竖排单字按x坐标聚类重建。

    :param page: PyMuPDF页面对象
    :param bbox: 图形bbox
    :return: (标签列表, 图形内水平线段)
    """
    x0, y0, x1, y1 = bbox
    page_h = page.rect.height
    vertical_chars: List[Tuple[float, float, str]] = []
    horizontal: List[Label] = []
    data = page.get_text("dict")
    for block in data.get("blocks", []):
        if block.get("type") != 0:
            continue
        for ln in block.get("lines", []):
            bx0, by0, bx1, by1 = ln["bbox"]
            cx, cy = (bx0 + bx1) / 2, (by0 + by1) / 2
            if not (x0 - 4 <= cx <= x1 + 4 and y0 - 4 <= cy <= y1 + 4):
                continue
            if by0 < page_h * 0.06 or by1 > page_h * 0.95:
                continue
            text = "".join(sp.get("text", "") for sp in ln.get("spans", [])).strip()
            if not text:
                continue
            # 竖排：整行只有1个字符且行框高≈宽（中文单字）
            if len(text) == 1 and _CJK_CHAR.match(text) and abs(
                    (by1 - by0) - (bx1 - bx0)) < 6:
                vertical_chars.append((bx0, (by0 + by1) / 2, text))
            else:
                horizontal.append(Label(text, (bx0, by0, bx1, by1)))
    # 竖排字符按x聚类（容差4pt），类内按y排序拼字
    labels = list(horizontal)
    used = [False] * len(vertical_chars)
    columns: List[List[Tuple[float, float, str]]] = []
    order = sorted(range(len(vertical_chars)), key=lambda i: vertical_chars[i][0])
    for i in order:
        if used[i]:
            continue
        base_x = vertical_chars[i][0]
        group = [vertical_chars[i]]
        used[i] = True
        for j in order:
            if not used[j] and abs(vertical_chars[j][0] - base_x) <= 4:
                group.append(vertical_chars[j])
                used[j] = True
        columns.append(group)
    # 单字高约10.5pt；同一标签内字间距≈11pt，不同层级框之间垂直空隙≥30pt，
    # 故按y间隙20pt切分，避免“行政人事部”与其下方“武汉销售处”串成一列
    for group in columns:
        group.sort(key=lambda c: c[1])
        runs: List[List[Tuple[float, float, str]]] = [[group[0]]]
        for cur in group[1:]:
            if cur[1] - runs[-1][-1][1] > 20:
                runs.append([cur])
            else:
                runs[-1].append(cur)
        for run in runs:
            text = "".join(c[2] for c in run)
            if len(text) >= 2:
                xs = [c[0] for c in run]
                ys = [c[1] for c in run]
                labels.append(Label(text, (min(xs) - 2, min(ys) - 6,
                                           max(xs) + 8, max(ys) + 6)))
    return labels, []


def _associate_text(inv, page_idx: int,
                    bbox: Tuple[float, float, float, float]
                    ) -> Tuple[str, str, str, str]:
    """在同页（必要时上一页）为图形关联图题/引导句/资料来源/邻近正文。

    :param inv: PdfInventory
    :param page_idx: 图形所在页0基下标
    :param bbox: 图形bbox
    :return: (图题, 引导句, 资料来源, 邻近正文)
    """
    x0, y0, x1, y1 = bbox
    caption, ref_sentence, source_line = "", "", ""
    nearby: List[str] = []
    title_re = re.compile(CONFIG.figure_title_pattern)
    ref_re = re.compile(CONFIG.figure_ref_pattern)
    # 无“图”字但紧邻图片的图表标题（如“2008年中国IC市场应用结构与增长(亿元)”）
    chart_title_re = re.compile(
        r"(增长|占比|构成|结构|趋势|份额|应用|市场).{0,12}"
        r"(\(|（|亿元|万元|％|%)")
    for ln in inv.pages_lines[page_idx]:
        tx0, ty0, tx1, ty1 = ln.bbox
        # 图题：图形上方75pt带内、含图类关键词的短行
        if (not caption and ty1 <= y0 + 8 and ty1 >= y0 - CONFIG.caption_band_pt
                and len(ln.text) <= 60 and title_re.search(ln.text)):
            caption = ln.text
        # 图形上方35pt内、不以句号结尾的图表特征短行
        if (not caption and ty1 <= y0 + 8 and ty1 >= y0 - 35
                and len(ln.text) <= 45 and chart_title_re.search(ln.text)
                and not re.search(r"[。；;]$", ln.text)):
            caption = ln.text
        # 图形下方紧邻的图题（部分图题压在图内顶部）
        if (not caption and ty0 >= y1 - 25 and ty0 <= y1 + CONFIG.caption_band_pt
                and len(ln.text) <= 60 and title_re.search(ln.text)):
            caption = ln.text
        if ref_re.search(ln.text) and len(ln.text) <= 80:
            ref_sentence = ln.text
        if _SOURCE_RE.search(ln.text) and ty0 > y1 - 10:
            source_line = ln.text
        # 邻近正文：图形上下130pt带内、版心内的正文行
        in_band = ((ty1 <= y0 + 8 and ty1 >= y0 - CONFIG.nearby_band_pt)
                   or (ty0 >= y1 - 8 and ty0 <= y1 + CONFIG.nearby_band_pt))
        if in_band and 60 < tx0 < 540 and not title_re.search(ln.text) \
                and not _SOURCE_RE.search(ln.text):
            nearby.append(ln.text)
    # 跨页引导句：上一页的“……如下图”句且该句下方当页再无图形
    if not ref_sentence and page_idx > 0:
        for ln in inv.pages_lines[page_idx - 1]:
            if not ref_re.search(ln.text):
                continue
            ref_y = ln.bbox[3]
            sub_rects, _ = _drawing_geometry(
                [d for d in inv.pages_drawings[page_idx - 1]
                 if d.get("rect") and d["rect"].y0 > ref_y + 6])
            sub_bbox = _union_bbox(sub_rects)
            if not (sub_bbox and sub_bbox[2] - sub_bbox[0] >= 150
                    and sub_bbox[3] - sub_bbox[1] >= 150):
                ref_sentence = ln.text
                break
    # 去重保序
    uniq: List[str] = []
    for t in nearby:
        if t not in uniq and t != caption:
            uniq.append(t)
    return caption, ref_sentence, source_line, " ".join(uniq[:8])


def extract_raster_figures(inv, image_dir: str, ocr: OcrEngine,
                           ocr_cache: dict) -> List[FigureRecord]:
    """抽取文档内全部有效嵌入位图并完成关联与OCR。

    :param inv: PdfInventory
    :param image_dir: 图片输出目录
    :param ocr: OCR引擎
    :param ocr_cache: OCR持久化缓存
    :return: 位图FigureRecord列表
    """
    # 先统计每个xref跨多少页出现，用于过滤页眉水印
    reuse: Dict[int, set] = {}
    size_info: Dict[int, Tuple[int, int]] = {}
    for idx, images in enumerate(inv.pages_images):
        for im in images:
            xref, w, h = im[0], im[2], im[3]
            reuse.setdefault(xref, set()).add(idx)
            size_info[xref] = (w, h)

    records: List[FigureRecord] = []
    seen_placements = set()
    for idx in range(inv.page_count):
        page = inv.page(idx)
        for im in inv.pages_images[idx]:
            xref, w, h = im[0], im[2], im[3]
            ratio = w / max(h, 1)
            if (xref in seen_placements
                    or len(reuse.get(xref, set())) > CONFIG.img_max_reuse_pages
                    or w < CONFIG.img_min_width or h < CONFIG.img_min_height
                    or w * h < CONFIG.img_min_area
                    or not (CONFIG.img_min_ratio <= ratio <= CONFIG.img_max_ratio)):
                continue
            seen_placements.add(xref)
            rects = page.get_image_rects(xref)
            if not rects:
                continue
            bbox = (rects[0].x0, rects[0].y0, rects[0].x1, rects[0].y1)
            fig_id = f"{inv.doc_tag}_p{idx + 1}_x{xref}"
            img_path = os.path.join(image_dir, f"{fig_id}.png")
            try:
                pix = pymupdf.Pixmap(inv.doc, xref)
                if pix.n > 4:
                    pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
                pix.save(img_path)
                thumb = _save_thumb(img_path)
            except Exception as exc:
                logger.warning("图片保存失败 %s：%s", fig_id, exc)
                continue
            caption, ref, source, nearby = _associate_text(inv, idx, bbox)
            labels, _ = _inner_labels(page, bbox)
            rec = FigureRecord(fig_id, inv.doc_tag, idx + 1, "raster", bbox,
                               image_path=img_path, thumb_path=thumb, xref=xref,
                               caption=caption, ref_sentence=ref,
                               source_line=source, nearby_text=nearby,
                               labels=labels, degraded=not ocr.available)
            rec.ocr_items = ocr.recognize(img_path, ocr_cache)
            records.append(rec)
            logger.info("位图：%s 第%d页 %dx%d 图题=%s",
                        inv.doc_tag, idx + 1, w, h, caption or "-")
    return records


def extract_vector_figures(inv, image_dir: str, ocr: OcrEngine,
                           ocr_cache: dict,
                           raster_records: List[FigureRecord]) -> List[FigureRecord]:
    """按图注门控抽取矢量结构图/流程图并渲染为PNG。

    :param inv: PdfInventory
    :param image_dir: 图片输出目录
    :param ocr: OCR引擎
    :param ocr_cache: OCR持久化缓存
    :param raster_records: 同页位图记录（大面积重叠时跳过，避免重复）
    :return: 矢量图FigureRecord列表
    """
    title_re = re.compile(CONFIG.figure_title_pattern)
    ref_re = re.compile(CONFIG.figure_ref_pattern)
    gated: Dict[int, str] = {}  # 页下标 -> 门控原因
    for idx in range(inv.page_count):
        page_text = "\n".join(ln.text for ln in inv.pages_lines[idx])
        reason = ""
        if title_re.search(page_text):
            reason = "图题门控"
        elif ref_re.search(page_text) and len(inv.pages_drawings[idx]) >= \
                CONFIG.figure_min_drawings:
            reason = "引导句门控"
        if reason:
            gated[idx] = reason
        # 跨页：当页出现“如下图”引导句、但当页引导句下方无图（版面不足
        # 图被排到次页），且次页存在足够大的绘图簇时，门控次页
        ref_below = False
        for ln in inv.pages_lines[idx]:
            if not ref_re.search(ln.text):
                continue
            ref_y = ln.bbox[3]
            # 当页是否已有嵌入位图落在引导句下方
            for rr in raster_records:
                if rr.page_no == idx + 1 and rr.bbox[1] > ref_y + 6:
                    ref_below = True
                    break
            # 当页引导句下方是否已有足够大的矢量绘图簇
            if not ref_below:
                sub_rects, _ = _drawing_geometry(
                    [d for d in inv.pages_drawings[idx]
                     if d.get("rect") and d["rect"].y0 > ref_y + 6])
                sub_bbox = _union_bbox(sub_rects)
                if sub_bbox and sub_bbox[2] - sub_bbox[0] >= 150 \
                        and sub_bbox[3] - sub_bbox[1] >= 150:
                    ref_below = True
            if ref_below:
                break
        nxt = idx + 1
        if (not ref_below and ref_re.search(page_text)
                and nxt < inv.page_count
                and len(inv.pages_drawings[nxt]) >= CONFIG.figure_min_drawings
                and nxt not in gated):
            nxt_rects, _ = _drawing_geometry(inv.pages_drawings[nxt])
            nxt_bbox = _union_bbox(nxt_rects)
            if nxt_bbox and nxt_bbox[2] - nxt_bbox[0] >= 150 \
                    and nxt_bbox[3] - nxt_bbox[1] >= 150:
                gated[nxt] = "跨页引导句门控"

    records: List[FigureRecord] = []
    seq = 0
    for idx, reason in gated.items():
        drawings = inv.pages_drawings[idx]
        if len(drawings) < CONFIG.figure_min_drawings:
            continue
        rects, hlines = _drawing_geometry(drawings)
        if not rects:
            continue
        bbox = _union_bbox(rects)
        if bbox is None:
            continue
        bx0, by0, bx1, by1 = bbox
        if (bx1 - bx0) < 150 or (by1 - by0) < 150:
            continue
        # 与同页位图大面积重叠则跳过（该图已有嵌入位图版本）
        overlap_skip = False
        for rr in raster_records:
            if rr.page_no != idx + 1:
                continue
            ax0, ay0, ax1, ay1 = rr.bbox
            ix = max(0, min(bx1, ax1) - max(bx0, ax0))
            iy = max(0, min(by1, ay1) - max(by0, ay0))
            area_r = max((ax1 - ax0) * (ay1 - ay0), 1)
            if ix * iy / area_r > 0.6:
                overlap_skip = True
                break
        if overlap_skip:
            continue
        seq += 1
        page = inv.page(idx)
        fig_id = f"{inv.doc_tag}_p{idx + 1}_v{seq}"
        img_path = os.path.join(image_dir, f"{fig_id}.png")
        clip = pymupdf.Rect(bx0 - 6, by0 - 6, bx1 + 6, by1 + 6)
        pix = page.get_pixmap(matrix=pymupdf.Matrix(CONFIG.vector_render_zoom,
                                                    CONFIG.vector_render_zoom),
                              clip=clip)
        pix.save(img_path)
        thumb = _save_thumb(img_path)
        caption, ref, source, nearby = _associate_text(
            inv, idx, (bx0, by0, bx1, by1))
        labels, _ = _inner_labels(
            page, (bx0 - 6, by0 - 6, bx1 + 6, by1 + 6))
        inner_h = [(y, xa, xb) for (y, xa, xb) in hlines
                   if by0 - 10 <= y <= by1 + 10]
        rec = FigureRecord(fig_id, inv.doc_tag, idx + 1, "vector",
                           (bx0, by0, bx1, by1), image_path=img_path,
                           thumb_path=thumb, caption=caption, ref_sentence=ref,
                           source_line=source, nearby_text=nearby,
                           labels=labels, h_lines=inner_h,
                           degraded=not ocr.available)
        rec.ocr_items = ocr.recognize(img_path, ocr_cache)
        records.append(rec)
        logger.info("矢量图：%s 第%d页（%s）标签%d个 图题=%s",
                    inv.doc_tag, idx + 1, reason, len(labels), caption or "-")
    return records


def extract_figures(inv, image_dir: str, ocr: OcrEngine,
                    ocr_cache: dict) -> List[FigureRecord]:
    """抽取单文档全部图像证据（位图+矢量图）。

    :param inv: PdfInventory
    :param image_dir: 图像库目录
    :param ocr: OCR引擎
    :param ocr_cache: OCR持久化缓存
    :return: 全部FigureRecord（位图在前、矢量图在后）
    """
    os.makedirs(image_dir, exist_ok=True)
    raster = extract_raster_figures(inv, image_dir, ocr, ocr_cache)
    vector = extract_vector_figures(inv, image_dir, ocr, ocr_cache, raster)
    return raster + vector
