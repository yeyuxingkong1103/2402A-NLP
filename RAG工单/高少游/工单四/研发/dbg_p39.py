# -*- coding: utf-8 -*-
"""核实 p39 区域 bbox 与 _node_boxes 实际纳入的节点。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pymupdf
from src import config
from src.figure_extractor import extract_figures_from_pdf
from src.figure_semantics import _node_boxes, _build_tree, _region_segments

figs = extract_figures_from_pdf(config.DATA_DIR / "招股说明书2.pdf")
p39 = [f for f in figs if f.page == 39]
for f in p39:
    print("p39 bbox:", f.bbox, "kind:", f.kind, "n_boxes:", f.n_boxes,
          "n_lines:", f.n_lines, "texts:", len(f.texts))

doc = pymupdf.open(str(config.DATA_DIR / "招股说明书2.pdf"))
page = doc.load_page(38)
for f in p39:
    boxes = _node_boxes(page, f.bbox)
    print("_node_boxes:", len(boxes))
    for b, n in sorted(boxes, key=lambda x: (x[0][1], x[0][0])):
        print(f"   {n:20s} ({b[0]:.1f},{b[1]:.1f},{b[2]:.1f},{b[3]:.1f})")
    segs = _region_segments(page, f.bbox)
    print("segs:", len(segs))
    root, ch = _build_tree(boxes, segs)
    print("root:", root)
    for k, v in ch.items():
        print(f"   {k} -> {v}")