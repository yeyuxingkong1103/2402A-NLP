# -*- coding: utf-8 -*-
"""工单14 问题集与实证产物的核对（只读）。

工单编号：人工智能NLP-RAG-修复低质量工业PDF 的解析与信息丢失工单
"""
from __future__ import annotations

import json
import pathlib

ROOT = pathlib.Path(r"E:\gao6gongdan\工单14")
Q = ROOT / "测试" / "测试数据" / "测试问题集.jsonl"
EV = ROOT / "研发" / "data" / "解析实证.json"

rows = [json.loads(l) for l in Q.read_text(encoding="utf-8").splitlines() if l.strip()]
print(f"题数: {len(rows)}")
for i, x in enumerate(rows, 1):
    g = x.get("group")
    print(f"--- 第{i}题 (group={g})")
    print("   问:", x["question"][:72])
    opts = x.get("options") or []
    for o in opts:
        print("     ", o[:80])
    print("   答案键:", x.get("answer"))

print()
ev = json.loads(EV.read_text(encoding="utf-8"))
print("实证 json 顶层键:", list(ev.keys()))
p1 = ev["1_文本层探针"]
print("文本层:", p1["verdict"], "| 字符数", p1["text_chars_total"], "| 页数", p1["pages"])
cov = ev["4_标记覆盖"]
print("标记恢复:", cov["recovered_count"], "/", cov["needed_count"])
print("  已恢复:", cov["recovered"])
print("  缺失  :", cov["missing"])
for k, v in ev["3_像素OCR"].items():
    print(f"  {k}: 命中 {v['hits_scored']} 字符 / {len(v['tokens'])} token / {v['elapsed_ms']} ms")
