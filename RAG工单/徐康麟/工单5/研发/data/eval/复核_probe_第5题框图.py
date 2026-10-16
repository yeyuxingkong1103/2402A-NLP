# -*- coding: utf-8 -*-
"""
工单：人工智能NLP-RAG-图像内容解析及检索优化
文件：研发/data/eval/复核_probe_第5题框图.py（T3，第 5 题组织结构图结构复核）

用途：
  1) 打印 PDF2 物理页 39 全部 span（文本 + bbox），按 y 行聚类，检查「销售部 / 大客户销售部 / 6 个销售处 / 上海分公司」的层级；
  2) 按 x 列聚类复原竖排文字（T1 规则：±6pt 列聚类 + 列内 y 升序）；
  3) 渲染页 39 全页 2x 图，供逐图视觉复核。
只读 PDF；图落盘到 研发/data/eval/复核_证据图/。
运行：pwsh -NoProfile -File run_py.ps1 研发/data/eval/复核_probe_第5题框图.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parents[3]
PDF2 = ROOT / "研发" / "data" / "raw" / "招股说明书2.pdf"
OUTDIR = ROOT / "研发" / "data" / "eval" / "复核_证据图"


def spans_of(page: fitz.Page):
    out = []
    for b in page.get_text("dict")["blocks"]:
        if b.get("type") != 0:
            continue
        for ln in b.get("lines", []):
            for sp in ln.get("spans", []):
                t = sp["text"]
                if t.strip():
                    out.append({"t": t, "bbox": [round(v, 1) for v in sp["bbox"]]})
    return out


def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    with fitz.open(PDF2) as doc:
        page = doc[38]  # 物理页 39
        print(f"页面尺寸 = {page.rect}")
        spans = spans_of(page)
        print(f"span 总数 = {len(spans)}")

        # 1) 按 y 行聚类打印
        spans_sorted = sorted(spans, key=lambda s: (round((s['bbox'][1] + s['bbox'][3]) / 2, 0), s['bbox'][0]))
        print("\n===== 按 y 行聚类（行内按 x 升序） =====")
        cur_y, line = None, []
        for s in spans_sorted:
            y = round((s["bbox"][1] + s["bbox"][3]) / 2, 0)
            if cur_y is None or abs(y - cur_y) > 6:
                if line:
                    print(f"  y≈{cur_y:6.0f} | " + " || ".join(f"{x['t']}@{x['bbox'][0]:.0f}" for x in line))
                cur_y, line = y, []
            line.append(s)
        if line:
            print(f"  y≈{cur_y:6.0f} | " + " || ".join(f"{x['t']}@{x['bbox'][0]:.0f}" for x in line))

        # 2) 按 x 列聚类（±6pt）
        cols: list[dict] = []
        for s in sorted(spans, key=lambda s: s["bbox"][0]):
            x0 = s["bbox"][0]
            placed = False
            for c in cols:
                if abs(c["x"] - x0) <= 6:
                    c["items"].append(s)
                    placed = True
                    break
            if not placed:
                cols.append({"x": x0, "items": [s]})
        print("\n===== 按 x 列聚类（±6pt，列内 y 升序拼接）—— 只为核对竖排标签 =====")
        for c in sorted(cols, key=lambda c: c["x"]):
            txt = "".join(i["t"] for i in sorted(c["items"], key=lambda i: i["bbox"][1]))
            if len(c["items"]) >= 2 or len(txt) > 1:
                ys = [round(i["bbox"][1]) for i in c["items"]]
                print(f"  x≈{c['x']:6.0f} y[{min(ys)}..{max(ys)}] n={len(c['items']):2d} -> {txt}")

        # 3) 渲染 2x 全页图
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        png = OUTDIR / "p39_组织结构图_2x.png"   # 与 构建_golden_qa.py 的产物同名，避免重复图
        pix.save(png)
        print(f"\n[OK] 渲染：{png}  ({pix.width}x{pix.height})")

        # 4) 落盘 span JSON
        (ROOT / "研发" / "data" / "eval" / "复核_probe_第5题_p39_spans.json").write_text(
            json.dumps(spans, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
