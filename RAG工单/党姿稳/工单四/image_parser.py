# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-图像内容解析及检索优化
图像解析模块：
  1. 用 PyMuPDF 定位"含大图/图表"的页面（图块面积占比超过阈值）；
  2. 将页面渲染为图片；
  3. 用 RapidOCR 提取图内文字（含图表中的坐标轴标签与数值），生成"图像语义文本块"；
  4. 过滤水印噪声，输出 type='image' 的块，参与后续检索与问答。
"""
import os
import re
import json
import fitz  # PyMuPDF
from config import PDF_PATHS, IMAGE_DIR, IMAGE_CHUNKS_FILE, FIGURE_AREA_RATIO, RENDER_ZOOM

_ocr = None
_WATERMARK = re.compile(r'八维|rengongzhi|人工智能NLP|维教育|智刘|能刘')


def get_ocr():
    """RapidOCR 单例（纯 ONNX，CPU 可用）"""
    global _ocr
    if _ocr is None:
        from rapidocr_onnxruntime import RapidOCR
        _ocr = RapidOCR()
    return _ocr


_FIG_CAPTION = re.compile(r'如下图|结构图|增长图|应用结构|示意图|流程图|趋势图|占比图|图\s*\d')


def find_figure_pages(doc):
    """
    判定"含图页"，满足任一条件即认为该页有图：
      1) 存在占页面面积 > FIGURE_AREA_RATIO 的位图；
      2) 矢量绘制元素数量 > DRAWING_THRESHOLD（组织结构图/饼图等矢量图）；
      3) 页面正文含图题关键字（如下图/结构图/增长图…）。
    """
    pages = []
    for i, page in enumerate(doc, 1):
        area = abs(page.rect.width * page.rect.height) or 1
        is_fig = False
        for im in page.get_image_info():
            x0, y0, x1, y1 = im["bbox"]
            if abs((x1 - x0) * (y1 - y0)) > FIGURE_AREA_RATIO * area:
                is_fig = True
                break
        if not is_fig:
            try:
                is_fig = len(page.get_drawings()) > DRAWING_THRESHOLD
            except Exception:
                is_fig = False
        if not is_fig and _FIG_CAPTION.search(page.get_text()):
            is_fig = True
        if is_fig:
            pages.append(i)
    return pages


def ocr_page_image(png_path):
    """对整页图片做 OCR，返回清洗后的文本"""
    res, _ = get_ocr()(png_path)
    lines = []
    for item in (res or []):
        txt = item[1].strip()
        if not txt or _WATERMARK.search(txt):
            continue
        lines.append(txt)
    # 去掉与页面正文完全重复的行不行，这里仅做去重相邻
    dedup = []
    for ln in lines:
        if not dedup or ln != dedup[-1]:
            dedup.append(ln)
    return "\n".join(dedup)


def extract_image_chunks(doc_name, pdf_path):
    """对单个PDF：找图页 -> 渲染 -> OCR -> 生成图像块"""
    os.makedirs(IMAGE_DIR, exist_ok=True)
    doc = fitz.open(pdf_path)
    fig_pages = find_figure_pages(doc)
    print(f"[图像解析] {doc_name}: 含图页 {len(fig_pages)} 页 {fig_pages[:20]}")
    chunks = []
    for pno in fig_pages:
        page = doc[pno - 1]
        pix = page.get_pixmap(matrix=fitz.Matrix(RENDER_ZOOM, RENDER_ZOOM))
        png = os.path.join(IMAGE_DIR, f"{doc_name}_p{pno}.png")
        pix.save(png)
        text = ocr_page_image(png)
        if len(text) < 20:
            continue
        # 取页面正文首行作为图题线索
        cap = page.get_text().strip().split("\n")[0][:60]
        chunks.append({
            "text": f"【图像·第{pno}页】{cap}\n{text}",
            "page": pno, "doc": doc_name, "type": "image", "image": png,
        })
    doc.close()
    return chunks


def build_image_chunks(save_path=IMAGE_CHUNKS_FILE):
    all_chunks = []
    for doc_name, path in PDF_PATHS.items():
        all_chunks.extend(extract_image_chunks(doc_name, path))
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    print(f"[图像解析] 生成图像块 {len(all_chunks)} 个，已保存: {save_path}")
    return all_chunks


def load_image_chunks(save_path=IMAGE_CHUNKS_FILE):
    with open(save_path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    cs = build_image_chunks()
    for c in cs:
        if c["page"] in (38, 39, 72):
            print(f"\n--- 图像块 第{c['page']}页 ---\n{c['text'][:400]}")
