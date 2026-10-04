# -*- coding: utf-8 -*-
"""
图像内容解析模块（工单04）
工单编号：人工智能NLP-RAG-图像内容解析及检索优化
说明：招股书中的组织结构图/市场图表多为矢量绘制（get_images 抓不到），
     采用"整页渲染 → 多模态模型(Qwen2.5-VL)语义解析"方案。
"""
import os
import fitz  # PyMuPDF


def find_pages_by_keyword(pdf_path, keywords):
    """定位包含指定关键词的页码（1起）"""
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        t = page.get_text()
        if any(kw in t for kw in keywords):
            pages.append(i + 1)
    doc.close()
    return pages


def render_pages(pdf_path, page_numbers, out_dir, zoom=2.0):
    """把指定页渲染为 PNG（整页截图，保留矢量图形内容）
    返回: [{"page": 页码, "png_path": 图片路径}, ...]
    """
    os.makedirs(out_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    out = []
    for pno in page_numbers:
        page = doc[pno - 1]
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        path = os.path.join(out_dir, f"p{pno}.png")
        pix.save(path)
        out.append({"page": pno, "png_path": path})
    doc.close()
    return out
