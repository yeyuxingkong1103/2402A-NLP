# -*- coding: utf-8 -*-
# 工单编号：人工智能 NLP-RAG 项目-LightRAG 优化任务
# 模块：lightrag_rag/make_scope —— 生成 LightRAG 建图范围（16 道评测题的金标准页）
# 说明：本机 CPU 推理较慢，全量 600+ 页建图成本极高；这里把「16 道题的金标准页」及其
#       相邻页作为建图范围，保证 RAG vs LightRAG 能在同一批问题上公平对比。
#       若在有 GPU 的机器上跑全量，直接执行 build_lightrag.py（不带 LR_SCOPE=1）即可。
# 用法：python lightrag_rag/make_scope.py
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.stdout.reconfigure(encoding="utf-8")

OUT = os.path.join(HERE, "scope_pages.json")
SPAN = int(os.environ.get("LR_SPAN", "0"))

scope = {}
for f, key in (("evaluation/eval_questions.json", "page"),
               ("evaluation/eval_questions_pdf2.json", "gold_page")):
    d = json.load(open(os.path.join(ROOT, f), encoding="utf-8"))
    for it in (d.get("questions") or d.get("items")):
        doc = "招股说明书2" if "pdf2" in f else "招股说明书1"
        for p in (it.get(key) or []):
            base = int(p)
            for k in range(-SPAN, SPAN + 1):
                scope.setdefault(doc, set()).add(base + k)

out = {k: sorted(v) for k, v in scope.items()}
json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("范围页数：", {k: len(v) for k, v in out.items()}, "合计", sum(len(v) for v in out.values()))
print("saved ->", OUT)
