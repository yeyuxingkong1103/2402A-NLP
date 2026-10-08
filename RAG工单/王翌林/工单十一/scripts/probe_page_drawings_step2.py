# -*- coding: utf-8 -*-
# 工单四 Step 2 验收辅助：探查目标页的位图/矢量图元构成
# 工单编号：人工智能NLP-RAG-图像内容解析及检索优化
import pymupdf

doc = pymupdf.open("../附件/招股说明书2.pdf")
for pno in [37, 38, 71, 72]:  # 1-based 页码
    page = doc.load_page(pno - 1)
    imgs = page.get_images(full=True)
    drawings = page.get_drawings()
    text = page.get_text()
    print(f"p{pno}: 位图数={len(imgs)} 矢量图元={len(drawings)} "
          f"文字数={len(text)} 含'组织结构'={'组织结构' in text}")
    for x in imgs:
        print("   xref", x[0], f"{x[2]}x{x[3]}")
doc.close()
