# -*- coding: utf-8 -*-
"""工单14 演示：扫描件专利 PDF 的解析链路实证。

工单编号：人工智能NLP-RAG-修复低质量工业PDF 的解析与信息丢失工单

本脚本做三件事，**全部落盘可追溯**：
  1. 文本层探针 —— 证明该 PDF 是纯扫描件（文本 0 字符、每页 1 张整页位图）
  2. 像素级附图标记识别 —— 复用 `研发/pipeline/font_ocr.py` 的自研方案
     （字体模板匹配 + 连通行/列投影，**不依赖任何 OCR 库**），在附图页上恢复
     附图标记（数字/字母/带撇号的 6'、6''、X1、h1…）及其坐标
  3. 与工单 6 题所需的标记做对照，输出「哪些标记被自动恢复、哪些缺失」

产出：`研发/data/解析实证.json`（供报告与截图引用）

用法：
  & $py 研发/scripts/演示_扫描件解析实证.py [--pages 2,7]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pymupdf
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "研发" / "pipeline"))

PDF = Path(r"E:\gao6gongdan\_共享输入\original_problems\documents\CN100342976C.pdf")
RENDER = ROOT / "研发" / "data" / "_render"
OUT = ROOT / "研发" / "data" / "解析实证.json"

# 6 题答案解析所依赖的附图标记（问题集在 工单14/_tmp_questions_cn100342976.jsonl）
NEEDED_MARKS = ["4", "5", "6", "6'", "6''", "X1", "X2", "X3", "h1", "h2", "D", "7", "10", "1"]


def probe_text_layer(doc) -> dict:
    """证明是不是纯扫描件。"""
    per_page = []
    for i, page in enumerate(doc):
        t = page.get_text().strip()
        imgs = page.get_images(full=True)
        per_page.append({
            "page": i + 1,
            "text_chars": len(t),
            "images": len(imgs),
            "rect": [round(page.rect.width, 1), round(page.rect.height, 1)],
        })
    total = sum(p["text_chars"] for p in per_page)
    return {
        "pages": doc.page_count,
        "text_chars_total": total,
        "per_page": per_page,
        "verdict": "纯扫描件（无文本层，必须走图像/视觉路径）" if total < 500 else "含文本层",
    }


def render_pages(doc, dpi: int = 150) -> list[dict]:
    RENDER.mkdir(parents=True, exist_ok=True)
    out = []
    for i, page in enumerate(doc):
        f = RENDER / f"page{i + 1:02d}.png"
        if not f.exists():
            page.get_pixmap(dpi=dpi).save(f)
        out.append({"page": i + 1, "file": str(f), "bytes": f.stat().st_size})
    return out


def ocr_marks(pages: list[int]) -> dict:
    """在指定页上跑自研像素 OCR，恢复附图标记。"""
    from font_ocr import FontTemplateOCR, binarize, extract_numerals, merge_into_tokens

    ocr = FontTemplateOCR()
    results = {}
    for pno in pages:
        f = RENDER / f"page{pno:02d}.png"
        if not f.exists():
            continue
        gray = np.asarray(Image.open(f).convert("L"))
        mask = binarize(gray, invert=False)
        t0 = time.perf_counter()
        hits = extract_numerals(gray, ocr)
        tokens = merge_into_tokens(hits)
        ms = (time.perf_counter() - t0) * 1000
        results[f"page{pno}"] = {
            "image": f.name,
            "hits_scored": len(hits),
            "tokens": [
                {"text": t.get("text"), "score": round(float(t.get("score", 0)), 4),
                 "x": round(float(t.get("x", 0)), 1), "y": round(float(t.get("y", 0)), 1)}
                for t in tokens
            ],
            "elapsed_ms": round(ms, 1),
        }
    return results


def coverage(ocr_results: dict, pages: list[int]) -> dict:
    """统计 6 题所需标记的恢复情况。"""
    found = {}
    for key in ocr_results:
        for t in ocr_results[key]["tokens"]:
            txt = str(t["text"])
            found.setdefault(txt, []).append({"page": key, "x": t["x"], "y": t["y"], "score": t["score"]})
    hit = [m for m in NEEDED_MARKS if any(m == k or k.startswith(m) for k in found)]
    miss = [m for m in NEEDED_MARKS if m not in hit]
    return {
        "needed": NEEDED_MARKS,
        "recovered": sorted(hit),
        "missing": sorted(miss),
        "recovered_count": len(hit),
        "needed_count": len(NEEDED_MARKS),
        "detail": found,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", default="2,7", help="要跑 OCR 的页码（逗号分隔）")
    args = ap.parse_args()
    pages = [int(x) for x in args.pages.split(",") if x.strip()]

    if not PDF.exists():
        print(f"[!!] 找不到 PDF: {PDF}")
        return 3

    doc = pymupdf.open(PDF)
    report = {
        "工单编号": "人工智能NLP-RAG-修复低质量工业PDF 的解析与信息丢失工单",
        "pdf": str(PDF),
        "pdf_bytes": PDF.stat().st_size,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "1_文本层探针": probe_text_layer(doc),
    }
    print(f"[1] 文本层探针：{report['1_文本层探针']['verdict']}"
          f"（共 {report['1_文本层探针']['pages']} 页，文本 {report['1_文本层探针']['text_chars_total']} 字符）")

    report["2_页面渲染"] = render_pages(doc)
    print(f"[2] 渲染 {len(report['2_页面渲染'])} 页 @150dpi -> {RENDER}")

    ocr_res = ocr_marks(pages)
    report["3_像素OCR"] = ocr_res
    for k, v in ocr_res.items():
        print(f"[3] {k}: 命中 {v['hits_scored']} 个字符、归并 {len(v['tokens'])} 个 token、耗时 {v['elapsed_ms']} ms")

    report["4_标记覆盖"] = coverage(ocr_res, pages)
    cov = report["4_标记覆盖"]
    print(f"[4] 所需标记恢复 {cov['recovered_count']}/{cov['needed_count']}；缺失：{cov['missing']}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] 实证产物 -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
