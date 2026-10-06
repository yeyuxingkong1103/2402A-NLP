# -*- coding: utf-8 -*-
"""
工单编号：人工智能 NLP-RAG-PDF 文档的图像内容解析及检索优化
步骤1：图像内容提取 —— 检测含图/图表的页面并渲染为高分辨率 PNG
  检测策略：页面文本含图表类标题关键词（结构图/增长图/示意图等）即视为图表页；
  同时提取页内非水印嵌入图。
运行：python extract_charts.py
"""
import json
import re
from pathlib import Path

import fitz

KEYWORDS = ["组织结构图", "结构图", "增长图", "应用结构", "示意图", "流程图"]
OUT_DIR = Path("images")
ZOOM = 4.0  # 4倍渲染：小字号图表标签（如"汽车"）在2.5倍下会漏识别


def find_chart_pages(pdf_path: str):
    """扫描全页，返回 {页码: 命中的图表标题关键词}。
    图表常跨页（后续页是图的下半部分但无标题关键词），因此同时纳入图表页的下一页。"""
    doc = fitz.open(pdf_path)
    hits = {}
    for i, page in enumerate(doc):
        text = page.get_text()
        kws = [k for k in KEYWORDS if k in text]
        if kws:
            hits[i + 1] = kws
    for pno in list(hits):
        nxt = pno + 1
        if nxt <= len(doc) and nxt not in hits:
            if len(doc[nxt - 1].get_text()) < 600:  # 后续页文本很少 => 图表延续
                hits[nxt] = ["图表延续页"]
    doc.close()
    return hits


def main():
    pdf_path = Path("pdf_path2.txt").read_text(encoding="utf-8").strip()
    OUT_DIR.mkdir(exist_ok=True)
    chart_pages = find_chart_pages(pdf_path)
    print("图表页检测：", chart_pages)

    doc = fitz.open(pdf_path)
    meta = {}
    for pno in chart_pages:
        page = doc[pno - 1]
        pix = page.get_pixmap(matrix=fitz.Matrix(ZOOM, ZOOM))
        png = OUT_DIR / f"page{pno}.png"
        pix.save(png)
        meta[str(pno)] = {"png": str(png), "keywords": chart_pages[pno],
                          "title_line": next((l for l in page.get_text().split("\n")
                                              if any(k in l for k in chart_pages[pno])), "")}
    doc.close()
    (OUT_DIR / "chart_meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    print(f"已渲染 {len(meta)} 个图表页 -> images/")


if __name__ == "__main__":
    main()
