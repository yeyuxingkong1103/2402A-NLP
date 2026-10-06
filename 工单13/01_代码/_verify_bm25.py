# -*- coding: utf-8 -*-
# 工单编号：人工智能 NLP-RAG 项目-RAG 性能瓶颈识别与优化
# 模块：evaluation/_verify_bm25 —— BM25 倒排优化「结果一致性 + 提速」自检
# 用法：python evaluation/_verify_bm25.py
import json
import os
import sys
import time
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "app"))
sys.stdout.reconfigure(encoding="utf-8")

import bm25 as B          # noqa: E402
from config import BM25_PATH  # noqa: E402

data = json.load(open(BM25_PATH, encoding="utf-8"))
tk = data["token_lists"]
new = B.BM25().fit_tokens(tk)

q = json.load(open(os.path.join(HERE, "eval_questions.json"), encoding="utf-8"))
items = q.get("questions") or q.get("items") or []
qs = [it.get("question") for it in items if it.get("question")]

# 旧算法（全库扫描）等价实现
old_scores = {}
t0 = time.perf_counter()
for s in qs:
    old_scores[s] = new._score_full(Counter(B.tokenize(s)))
t1 = time.perf_counter()
new_scores = {}
for s in qs:
    new_scores[s] = new.score(s)
t2 = time.perf_counter()

ok = all(old_scores[s] == new_scores[s] for s in qs)
print("题目数 =", len(qs), "| 逐位一致 =", ok)
print("旧：全库扫描  %.1f ms/次" % ((t1 - t0) * 1000 / len(qs)))
print("新：倒排表打分 %.2f ms/次" % ((t2 - t1) * 1000 / len(qs)))
