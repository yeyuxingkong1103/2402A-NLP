# -*- coding: utf-8 -*-
# 工单编号：人工智能 NLP-RAG-基于 PDF 文档的问答系统优化
# 模块：smoke_test —— 小样本端到端冒烟（3 页 MinerU 输出）
import os, sys, json
sys.stdout.reconfigure(encoding="utf-8")
APP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "app")
sys.path.insert(0, APP)

import parse as P
from clean import clean_blocks
from chunk import build_chunks
import kb as KB
import engine

TEST_CL = r"C:\Users\xzr\.openclaw\workspace\tmp_rag_proj\mineru_test\招股说明书1\txt\招股说明书1_content_list.json"
cl = json.load(open(TEST_CL, encoding="utf-8"))
blocks = P.content_list_to_blocks(cl)
print("blocks:", len(blocks), "| pages:", sorted({b['page'] for b in blocks}))
blocks = clean_blocks(blocks)
chunks = build_chunks(blocks)
print("chunks:", len(chunks))
for c in chunks[:3]:
    print("  p%s [%s] %s" % (c['page'], c['section'], c['text'][:70]))

km = KB.KB().build(chunks)
KB._CACHE["kb"] = km
print("dim:", km.emb.shape)
h, conf = engine.retrieve("每股发行价格是多少")
print("query 每股发行价格 -> conf=%.3f" % conf)
for x in h:
    print("   p%s sim=%.3f %s" % (x['page'], x['dense_sim'], x['text'][:60]))
print("OK smoke ✅")
