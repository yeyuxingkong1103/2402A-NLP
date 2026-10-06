# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-图像内容解析及检索优化
# 模块：tools/extract_figures —— 从 PDF 中抽取“有效图表”（过滤水印/装饰小图）
# 说明：PDF2 每页都嵌了 ~15 个小图（水印/装饰），直接取图会得到几千张垃圾。
#       本脚本按「占页面面积比 + 像素尺寸」双重过滤，只导出真正的图表，
#       并把页面文字层里与该图 bbox 重叠的文字一并记录（供 OCR/语义描述复用）。
# 输出：data/figures/<doc>/pXXX_k.png + data/figures/figures.json
import os
import sys
import json

sys.stdout.reconfigure(encoding="utf-8")
import pymupdf  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "data", "figures")
MIN_AREA_RATIO = 0.05     # 图面积 / 页面面积 下限
MIN_W, MIN_H = 120, 80    # 最小像素尺寸（页面坐标 1pt=1px @72dpi 口径）


def extract(pdf_path, doc_id, dpi=150):
    os.makedirs(os.path.join(OUT_DIR, doc_id), exist_ok=True)
    doc = pymupdf.open(pdf_path)
    figs = []
    for i, page in enumerate(doc, 1):
        parea = page.rect.width * page.rect.height
        try:
            infos = page.get_image_info(xrefs=True)
        except Exception:  # noqa: BLE001
            infos = []
        k = 0
        for info in infos:
            bbox = pymupdf.Rect(info.get("bbox") or (0, 0, 0, 0))
            if bbox.is_empty:
                continue
            if bbox.width < MIN_W or bbox.height < MIN_H:
                continue
            if (bbox.width * bbox.height) < MIN_AREA_RATIO * parea:
                continue
            # 页面文字层里与该图重叠的文字（图内文字常以文本对象存在，可直接复用）
            try:
                txt = page.get_text("text", clip=bbox).strip()
            except Exception:  # noqa: BLE001
                txt = ""
            k += 1
            name = "p%03d_%d.png" % (i, k)
            path = os.path.join(OUT_DIR, doc_id, name)
            try:
                page.get_pixmap(clip=bbox, dpi=dpi).save(path)
            except Exception:  # noqa: BLE001
                continue
            figs.append({"doc": doc_id, "page": i, "file": os.path.relpath(path, ROOT).replace("\\", "/"),
                         "bbox": [round(v, 1) for v in bbox], "w": round(bbox.width), "h": round(bbox.height),
                         "text_layer": txt[:800]})
    return figs


def main():
    pdfs = sys.argv[1:] or [os.path.join(ROOT, "data", "pdfs", "招股说明书2.pdf")]
    all_figs = []
    for p in pdfs:
        doc_id = os.path.splitext(os.path.basename(p))[0]
        figs = extract(p, doc_id)
        print("%-20s figures=%d" % (doc_id, len(figs)))
        all_figs += figs
    os.makedirs(OUT_DIR, exist_ok=True)
    json.dump(all_figs, open(os.path.join(OUT_DIR, "figures.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("total figures:", len(all_figs), "->", os.path.join(OUT_DIR, "figures.json"))


if __name__ == "__main__":
    main()
