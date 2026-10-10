# -*- coding: utf-8 -*-
"""调试：抽取并解析关键图形（组织结构图 / IC 市场增长图）。"""
import sys, logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

from src.figure_extractor import extract_all_figures
from src.figure_semantics import analyze_figures

figs = extract_all_figures()
print("图形区域总数:", len(figs))
for f in figs:
    print(f"  {f.key} kind={f.kind} boxes={f.n_boxes} lines={f.n_lines} "
          f"texts={len(f.texts)} cap={f.caption[:40]!r}")

print("=" * 70)
sems = analyze_figures(figs)
print("语义解析总数:", len(sems))
for s in sems:
    if s.relations or s.key_values:
        print("-" * 70)
        print("KEY:", s.key, "TYPE:", s.figure_type, "score:", s.type_score)
        print(s.semantic_text[:900])