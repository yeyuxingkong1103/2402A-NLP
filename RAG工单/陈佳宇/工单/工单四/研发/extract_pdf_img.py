# -*- coding:utf-8 -*-
"""
工单编号：人工智能NLP‑RAG‑图像内容解析及检索优化
功能：将PDF每一页渲染为整张PNG图片（兼容reportlab矢量绘图），保存至pdf_images目录
"""
import os
import pymupdf

PDF_FILE = "招股说明书2.pdf"
OUT_IMG_DIR = "pdf_images"

if not os.path.exists(OUT_IMG_DIR):
    os.makedirs(OUT_IMG_DIR)

doc = pymupdf.open(PDF_FILE)
img_count = 0

for page_idx, page in enumerate(doc):
    # 将PDF页面渲染为像素图片
    pix = page.get_pixmap()
    out_path = os.path.join(OUT_IMG_DIR, f"page{page_idx}.png")
    pix.save(out_path)
    img_count +=1
    print(f"保存页面图片：{out_path}")

doc.close()
print(f"\n✅ PDF页面渲染完成，共生成 {img_count} 张页面图片，保存在 {OUT_IMG_DIR}")
