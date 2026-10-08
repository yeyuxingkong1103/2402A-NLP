# -*- coding: utf-8 -*-
# 工单四 Step 2 验收辅助：定位两份 PDF 中"组织结构/IC市场"关键词页码
# 工单编号：人工智能NLP-RAG-图像内容解析及检索优化
import pymupdf

for pdf, kws in [
    ("../附件/招股说明书2.pdf", ["组织结构", "IC市场", "应用结构"]),
    ("../附件/招股说明书1.pdf", ["组织结构"]),
]:
    doc = pymupdf.open(pdf)
    print("==", pdf)
    for kw in kws:
        hits = []
        for p in range(doc.page_count):
            if kw in doc.load_page(p).get_text():
                hits.append(p + 1)
        print(f"  '{kw}' 页码: {hits[:15]}")
    doc.close()
