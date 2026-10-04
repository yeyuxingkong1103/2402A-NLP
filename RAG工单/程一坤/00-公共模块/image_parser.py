# -*- coding: utf-8 -*-
"""
图像内容解析模块（工单04）
工单编号：人工智能NLP-RAG-图像内容解析及检索优化
说明：招股书中的组织结构图/市场图表多为矢量绘制（get_images 抓不到），
     采用"整页渲染 → 多模态模型(Qwen2.5-VL)语义解析"方案。
"""
import os  # 创建输出目录、拼接图片保存路径
import fitz  # PyMuPDF：页面渲染为位图


def find_pages_by_keyword(pdf_path, keywords):
    """定位包含指定关键词的页码（1起）"""
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        t = page.get_text()  # 提取该页全文（含矢量文字）
        # 任一关键词命中即收录该页；用 in 子串匹配而非正则，简单可靠
        if any(kw in t for kw in keywords):
            pages.append(i + 1)  # i 从 0 起，+1 转为 1 起页码
    doc.close()
    return pages


def render_pages(pdf_path, page_numbers, out_dir, zoom=2.0):
    """把指定页渲染为 PNG（整页截图，保留矢量图形内容）
    返回: [{"page": 页码, "png_path": 图片路径}, ...]
    """
    # exist_ok=True：重复运行时不因目录已存在而报错
    os.makedirs(out_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    out = []
    for pno in page_numbers:
        # 传入的页码是 1 起，fitz 内部索引是 0 起，需 -1
        page = doc[pno - 1]
        # zoom=2.0 表示 2 倍分辨率渲染：太小则图中文字模糊，VL 模型识别不了；
        # Matrix(zoom, zoom) 构造缩放矩阵传给 get_pixmap
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        path = os.path.join(out_dir, f"p{pno}.png")
        pix.save(path)  # 保存位图；用页码命名，与 PDF 页一一对应
        out.append({"page": pno, "png_path": path})
    doc.close()
    return out
