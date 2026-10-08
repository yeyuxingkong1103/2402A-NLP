# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-图像内容解析及检索优化
# scripts/check_eval_rows_step5.py —— 分析评估 JSON：图像题答案与低延迟行（临时辅助）
import json
from pathlib import Path

r = json.loads(Path("docs/eval_v4_results.json").read_text(encoding="utf-8"))
print("== 图像题答案 ==")
for row in r["rows"]:
    if row["id"] in (105, 106):
        print(f"id{row['id']} [{row['mode']}] acc={row['accuracy']} "
              f"kws={row['kws_hit']} n_img={row['n_images']}")
        print(f"   {row['answer'][:150]}")
print("\n== 低延迟疑似兜底行 ==")
for row in r["rows"]:
    if row["mode"] in ("v3", "v4") and row["latency_ms"] < 200:
        print(f"id{row['id']} [{row['mode']}] {row['answer'][:50]}")
