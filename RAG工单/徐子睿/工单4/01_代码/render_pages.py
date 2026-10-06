# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-图像内容解析及检索优化
# 模块：tools/render_pages —— 整页渲染（覆盖矢量图形，如组织结构图）
# 说明：PDF 里很多“图”是矢量绘图（get_image_info 抓不到）。这里把页面整体渲染成图片，
#       配合 CLIP 做图文语义匹配，既覆盖位图表格/照片，也覆盖矢量流程图、组织结构图。
# 输出：data/figures/<doc>/pages/pXXX.png
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
import pymupdf  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "figures")


def render(pdf_path, doc_id, dpi=110):
    dst = os.path.join(OUT, doc_id, "pages")
    os.makedirs(dst, exist_ok=True)
    doc = pymupdf.open(pdf_path)
    n = 0
    for i, page in enumerate(doc, 1):
        p = os.path.join(dst, "p%03d.png" % i)
        if os.path.exists(p):
            n += 1
            continue
        page.get_pixmap(dpi=dpi).save(p)
        n += 1
    return n


if __name__ == "__main__":
    pdfs = sys.argv[1:] or [os.path.join(ROOT, "data", "pdfs", "招股说明书2.pdf")]
    for p in pdfs:
        d = os.path.splitext(os.path.basename(p))[0]
        print("%-20s pages rendered = %d" % (d, render(p, d)))
