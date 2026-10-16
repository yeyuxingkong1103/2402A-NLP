# -*- coding: utf-8 -*-
"""临时探针：查看图谱产物字段结构（工单编号：人工智能NLP-RAG-Graph RAG 优化任务）。"""
import io
import json
import sys

sys.stdout.reconfigure(encoding="utf-8")
P = "E:/gao6gongdan/工单8/研发/data/graph/"
for f in ["entities.jsonl", "relations.jsonl"]:
    print("=====", f)
    with io.open(P + f, encoding="utf-8") as h:
        for i, line in enumerate(h):
            if i >= 2:
                break
            print(json.dumps(json.loads(line), ensure_ascii=False)[:1000])
print("===== 图谱构建统计.json")
print(io.open(P + "图谱构建统计.json", encoding="utf-8").read()[:2000])
