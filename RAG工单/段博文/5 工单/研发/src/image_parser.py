# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-Query理解优化任务
"""
PDF 图像内容解析模块（多模态增强）：定位 PDF 中的结构图/统计图并生成可检索语义。

背景：
    招股说明书中的组织结构图、市场增长图有的是「矢量图形」，有的是「内嵌位图」，
    均不在文本层，普通 PDF 文本解析会完全丢失其中信息。

多模态处理流水线：
    1. 图像定位：图题/图引用关键词 + 矢量绘图密度/内嵌位图，判定哪些页含图；
    2. 图像获取：
       - 内嵌位图：逐张提取原始字节（一页多图时分开，避免数字相互混淆）；
       - 纯矢量图：整页高分辨率渲染成位图；
    3. 视觉识别：RapidOCR（深度视觉模型）识别图中中文文字、部门名、百分比数字；
    4. 语义生成：图题 + OCR 结果交给 LLM，生成结构化、事实完整的语义描述块；
    5. 统一入库：作为 block_type='image' 的 Document，由 bge-m3 向量化检索。

选型说明：本场景需精确读取「图中文字与数字」（数部门个数、读 -2.0% 等），CLIP
仅做图文跨模态对齐、中文读数能力弱，因此采用「OCR 视觉模型 + LLM」的多模态组合。
"""

import os
from typing import List, Tuple

import fitz  # PyMuPDF
import httpx
from langchain_core.documents import Document

from config import (
    IMAGE_RENDER_ZOOM, IMAGE_STROKE_MIN,
    IMAGE_BITMAP_MIN_W, IMAGE_BITMAP_MIN_H,
    IMAGE_MAX_PER_PDF, IMAGE_CAPTION_KEYWORDS, IMAGE_WATERMARKS,
    IMAGE_MIN_OCR_BOXES, IMAGE_MIN_OCR_CHARS, IMAGE_CHART_MIN_CHARS,
    LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, LLM_TEMPERATURE,
)
from logger import get_logger

logger = get_logger(__name__)

_ocr = None


def get_ocr():
    """获取 RapidOCR 单例（首次加载较慢）。"""
    global _ocr
    if _ocr is None:
        from rapidocr_onnxruntime import RapidOCR
        _ocr = RapidOCR()
        logger.info("多模态 OCR 视觉模型已加载（RapidOCR）")
    return _ocr


# ============================================================
# 一、图像页定位
# ============================================================
def _stroke_count(page) -> int:
    """统计页面矢量绘图的笔画总数（矩形、连线、曲线等）。"""
    total = 0
    try:
        for d in page.get_drawings():
            total += len(d.get("items", []))
    except Exception:
        return 0
    return total


def _is_real_bitmap(w: int, h: int) -> bool:
    """双维判定是否为真实图片（排除 1 像素线条色块与水印小图）。"""
    return w >= IMAGE_BITMAP_MIN_W and h >= IMAGE_BITMAP_MIN_H


def _has_large_bitmap(page) -> bool:
    """是否含有真实内嵌位图。"""
    try:
        for im in page.get_images(full=True):
            if _is_real_bitmap(im[2], im[3]):
                return True
    except Exception:
        return False
    return False


def _iter_real_bitmaps(page) -> List[Tuple[int, bytes, int, int]]:
    """逐张提取页面中的真实位图（按 xref 去重）。

    Returns:
        [(xref, image_bytes, width, height), ...]
    """
    out = []
    seen = set()
    try:
        for im in page.get_images(full=True):
            xref, w, h = im[0], im[2], im[3]
            if xref in seen or not _is_real_bitmap(w, h):
                continue
            seen.add(xref)
            info = page.parent.extract_image(xref)
            out.append((xref, info["image"], w, h))
    except Exception as e:
        logger.warning(f"提取位图失败：{e}")
    return out


def _caption_hit(text: str) -> bool:
    """文本是否命中图题/图引用关键词。"""
    return any(kw in text for kw in IMAGE_CAPTION_KEYWORDS)


def locate_figure_pages(pdf_path: str) -> List[Tuple[int, str]]:
    """定位含图页。

    规则（满足任一）：
        A. 本页含真实内嵌位图——图本身就是证据；
        B. 本页命中图引用词，且矢量笔画数达标（图与引用同页）；
        C. 上一页命中图引用词、本页笔画数达标（图在引用的下一页）。

    Returns:
        [(page_number, reason), ...]
    """
    doc = fitz.open(pdf_path)
    picked = []
    prev_ref = False
    for i, page in enumerate(doc, start=1):
        text = page.get_text()
        strokes = _stroke_count(page)
        ref = _caption_hit(text)
        has_bmp = _has_large_bitmap(page)

        reason = None
        if has_bmp:
            reason = f"内嵌大图(strokes={strokes})"
        elif ref and strokes >= IMAGE_STROKE_MIN:
            reason = f"图引用+矢量图(strokes={strokes})"
        elif prev_ref and strokes >= IMAGE_STROKE_MIN and not ref:
            reason = f"承接上页图引用(strokes={strokes})"

        if reason:
            picked.append((i, reason))
        prev_ref = ref
    doc.close()
    return picked[:IMAGE_MAX_PER_PDF]


# ============================================================
# 二、渲染 + OCR
# ============================================================
def _render_page(page) -> bytes:
    """将页面渲染为 PNG 字节。"""
    zoom = fitz.Matrix(IMAGE_RENDER_ZOOM, IMAGE_RENDER_ZOOM)
    pix = page.get_pixmap(matrix=zoom)
    return pix.tobytes("png")


def _is_watermark(text: str) -> bool:
    low = text.lower()
    return any(w.lower() in low for w in IMAGE_WATERMARKS)


def _ocr_image(img_bytes: bytes) -> List[str]:
    """OCR 识别，返回去水印后的文本（保留顺序）。"""
    ocr = get_ocr()
    result, _ = ocr(img_bytes)
    if not result:
        return []
    texts = []
    for item in result:
        txt = item[1].strip()
        if txt and not _is_watermark(txt):
            texts.append(txt)
    return texts


def _extract_caption(text: str) -> str:
    """从文本层提取可能的图题行（含'图'的较短行）。"""
    captions = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if ("图" in line) and len(line) <= 40:
            captions.append(line)
    return "；".join(captions[:3])


def _ocr_boxes(img_bytes: bytes) -> List[Tuple[list, str]]:
    """OCR 识别，返回去水印的 (box, text)，保留空间坐标。

    box = [[x0,y0],[x1,y1],[x2,y2],[x3,y3]]（图像像素坐标，左上原点）
    """
    ocr = get_ocr()
    result, _ = ocr(img_bytes)
    if not result:
        return []
    out = []
    for box, text, score in result:
        text = text.strip()
        if text and not _is_watermark(text):
            out.append((box, text))
    return out


def _box_center(box) -> Tuple[float, float]:
    xs = [p[0] for p in box]
    ys = [p[1] for p in box]
    return sum(xs) / 4, sum(ys) / 4


def group_by_row(boxes: List[Tuple[list, str]], img_height: float) -> List[List[Tuple[str, float]]]:
    """按中心 y 坐标把文本框聚成行，行内按 x 排序。

    用于统计图：同一水平线上的「行业名 + 百分比」自然归为一行，保证配对正确。

    Returns:
        [[(text, center_x), ...], ...]（行按 y 从上到下）
    """
    if not boxes:
        return []

    items = []
    for box, text in boxes:
        cx, cy = _box_center(box)
        items.append((cy, cx, text))
    items.sort(key=lambda t: t[0])

    y_tol = img_height * 0.025  # 行容差：图像高度的 2.5%
    rows = []
    cur = [items[0]]
    for it in items[1:]:
        if abs(it[0] - cur[-1][0]) <= y_tol:
            cur.append(it)
        else:
            rows.append(cur)
            cur = [it]
    rows.append(cur)

    out = []
    for r in rows:
        r.sort(key=lambda t: t[1])  # 行内从左到右
        out.append([(text, cx) for cy, cx, text in r])
    return out


def _rows_to_text(rows: List[List[Tuple[str, float]]]) -> str:
    """把行结构渲染成可读文本（每行用 ' <- ' 表示同属一行）。"""
    lines = []
    for r in rows:
        labels = [t for t, _ in r]
        lines.append(" / ".join(labels))
    return "\n".join(lines)


def _has_percent(text: str) -> bool:
    import re
    return bool(re.search(r"-?\d+(?:\.\d+)?%", text))


_PAIR_TITLE_WORDS = {
    "增长率", "增长", "规模", "市场规模", "亿元", "万元", "金额",
    "占比", "比例", "单位", "年份", "类别", "项目",
}


def pair_label_value(boxes: List[Tuple[list, str]], img_h: float) -> List[Tuple[str, str]]:
    """标签（行业名，左列）与数值（百分比，条右端）按 y 就近贪心配对。

    比"严格同行"更鲁棒：条形末端数字与左侧行业名可能有十几像素 y 偏差，
    用 y 最近匹配可正确归位。

    Returns:
        [(label, value), ...]（未配上的 label 记为 "(未配对)"）
    """
    import re
    labels, values = [], []
    for box, text in boxes:
        cx, cy = _box_center(box)
        if _has_percent(text):
            values.append((cy, cx, text))
        elif not re.search(r"\d{3,}", text) and text not in _PAIR_TITLE_WORDS:  # 排除金额与标题词
            labels.append((cy, cx, text))

    if not values:
        return []

    labels.sort()
    pairs = []
    used = set()
    tol = img_h * 0.08
    for vcy, vcx, vtext in values:
        best, bd = None, 1e18
        for i, (lcy, lcx, ltext) in enumerate(labels):
            if i in used:
                continue
            d = abs(vcy - lcy)
            if d < bd:
                bd, best = d, i
        if best is not None and bd <= tol:
            pairs.append((labels[best][2], vtext))
            used.add(best)
        else:
            pairs.append(("(未配对)", vtext))

    # 按数值排序，便于直接看最大/最小
    def val(p):
        m = re.search(r"-?\d+(?:\.\d+)?", p[1])
        return float(m.group()) if m else 0.0
    pairs.sort(key=val)
    return pairs


def pairs_to_text(pairs: List[Tuple[str, str]]) -> str:
    return "\n".join(f"{lab} / {val}" for lab, val in pairs)


def extract_text_hint(page_text: str) -> str:
    """从正文提取与图相关的事实约束句（含'增长'且含数字的句子）。"""
    hints = []
    for line in page_text.splitlines():
        s = line.strip()
        if s and ("增长" in s) and re_search_digit(s) and len(s) <= 120:
            hints.append(s)
    return "；".join(hints[:3])


def re_search_digit(s: str) -> bool:
    import re
    return bool(re.search(r"\d", s))


# ============================================================
# 三、LLM 语义生成
# ============================================================
def _llm_describe(caption: str, rows_text: str, hint: str) -> str:
    """把「按空间行对齐的 OCR 结果」+ 正文事实约束交给 LLM，生成语义描述。"""
    sys_prompt = (
        "你是一名严谨的招股说明书图表分析专家。系统已用 OCR 识别出图中文字，并按"
        "「同一水平线归为一行」的方式做了空间对齐（每行文字在图中处于同一水平位置）。"
        "请据此还原这张图的真实含义，输出准确、可被检索的中文语义描述。"
    )
    user_prompt = f"""【图题/上下文】{caption or '（未提取到）'}

【OCR按行对齐结果】（同一行内用 " / " 分隔，表示它们在图中处于同一水平线）
{rows_text}

【正文事实约束（必须与此一致）】
{hint or '（无）'}

【输出要求】
1. 判断图类型（组织结构图 / 柱状增长率图 / 饼图 / 流程图 / 股权结构图等）；
2. 配对规则（关键）：
   - 增长率柱状图：行业名与它的百分比在「同一行」，必须按行配对，严禁跨行猜测；
     逐一列出每个行业及其增长率，明确增长率最高者、最低者、唯一负增长者；
   - 组织结构图：列出出现的所有部门，明确"某部门下辖哪几个部门/销售处"并给出数量；
   - 饼图：列出各部分名称及占比；
3. 数字必须忠实于按行对齐结果与正文约束，不得编造；若按行结果与正文约束冲突，以正文约束为准并修正配对；
4. 忽略页眉页脚、水印；用通顺的中文分点陈述，确保关键名称与数字都出现。"""

    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": LLM_TEMPERATURE,
        "max_tokens": 900,
    }
    with httpx.Client(timeout=120.0) as client:
        resp = client.post(f"{LLM_BASE_URL}/chat/completions", headers=headers, json=payload)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


# ============================================================
# 四、对外主入口
# ============================================================
def extract_image_documents(pdf_path: str, file_name: str) -> List[Document]:
    """解析 PDF 中所有含图页，生成图像语义 Document 列表。"""
    if not os.path.isfile(pdf_path):
        logger.error(f"PDF 不存在：{pdf_path}")
        return []

    pages = locate_figure_pages(pdf_path)
    logger.info(f"[{file_name}] 定位到 {len(pages)} 个含图页：{[p[0] for p in pages]}")

    documents = []
    doc = fitz.open(pdf_path)
    block_seq = 0
    for pno, reason in pages:
        page = doc[pno - 1]
        page_text = page.get_text()
        base_caption = _extract_caption(page_text) or f"第{pno}页插图"
        hint = extract_text_hint(page_text)

        # 路径1：逐张真实内嵌位图分别 OCR（一页多图时数字不相互混淆）
        bitmaps = _iter_real_bitmaps(page)
        if bitmaps:
            multi = len(bitmaps) > 1
            targets = [
                (img_bytes, h, f"{base_caption}（图{j}）" if multi else base_caption)
                for j, (xref, img_bytes, w, h) in enumerate(bitmaps, 1)
            ]
        else:
            # 路径2：纯矢量结构图，整页高分辨率渲染，渲染高=页高*缩放
            targets = [(None, page.rect.height * IMAGE_RENDER_ZOOM, base_caption)]

        for img_bytes, img_h, caption in targets:
            try:
                raw = _render_page(page) if img_bytes is None else img_bytes
                boxes = _ocr_boxes(raw)
                if not boxes:
                    logger.warning(f"  第{pno}页「{caption}」OCR 无结果，跳过")
                    continue

                # 信息密度门槛：过滤照片/印章/Logo/凭证（OCR 文字稀少），只保留图表
                valid_texts = [t.strip() for _, t in boxes if len(t.strip()) >= 2]
                total_chars = sum(len(t) for t in valid_texts)
                has_pct = any(_has_percent(t) for _, t in boxes)
                char_require = IMAGE_MIN_OCR_CHARS if has_pct else IMAGE_CHART_MIN_CHARS
                if len(valid_texts) < IMAGE_MIN_OCR_BOXES or total_chars < char_require:
                    logger.info(
                        f"  第{pno}页「{caption}」OCR 信息不足（{len(valid_texts)}框/{total_chars}字，门槛{char_require}），判为非图表，跳过"
                    )
                    continue

                # 含百分比→标签/数值就近配对（统计图）；否则→按行对齐（结构图）
                if any(_has_percent(t) for _, t in boxes):
                    pairs = pair_label_value(boxes, img_h)
                    structured = pairs_to_text(pairs)
                else:
                    structured = _rows_to_text(group_by_row(boxes, img_h))

                semantic = _llm_describe(caption, structured, hint)

                content = f"【图像语义·{caption}】\n{semantic}"
                documents.append(
                    Document(
                        page_content=content,
                        metadata={
                            "source": file_name,
                            "page_number": pno,
                            "block_index": block_seq,
                            "block_type": "image",
                            "image_caption": caption,
                            "ocr_box_count": len(boxes),
                            "chunk_index": -1,
                        },
                    )
                )
                block_seq += 1
                logger.info(f"  第{pno}页「{caption}」语义已生成（OCR {len(boxes)} 框）")
            except Exception as e:
                logger.error(f"  第{pno}页「{caption}」处理失败：{e}")
                continue
    doc.close()
    return documents


def preview_figure_pages(pdf_path: str):
    """只做选页预览（不调用 OCR/LLM），便于人工核对选页是否合理。"""
    pages = locate_figure_pages(pdf_path)
    print(f"文件：{os.path.basename(pdf_path)}")
    print(f"共选中 {len(pages)} 个含图页：")
    for pno, reason in pages:
        print(f"  第 {pno} 页  [{reason}]")
    return pages
