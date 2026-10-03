# -*- coding: utf-8 -*-
# 工单编号：人工智能 NLP-RAG-基于 PDF 文档的问答系统优化
# 模块：tune_fusion —— 融合策略对比（RRF vs 归一化加权求和）
# 说明：金标准页码按“证据串出现在哪些页”自动展开（避免人工收窄），
#       对比多种融合配置的 Hit@1/@3/@5 与 MRR，辅助选定默认参数。
import os, sys, json
sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "app"))
import kb as KB, engine

EVAL = json.load(open(os.path.join(HERE, "eval_questions.json"), encoding="utf-8"))
km = KB.KB().load()
KG = KB._CACHE["kb"] = km

# 证据串：用于把金标准页码扩展为“所有包含答案证据的页”
EVIDENCE = {
    260: ["6,464.51", "14,414.16", "18,780.67", "4,627.14"],
    95: ["视频指挥系统技术标准", "某视频技术规范"],
    33: ["82.10%", "97.31%", "94.84%", "94.34%"],
    34: ["电子元器件制造企业"],
    957: ["已经成为国防军队视频指挥领域的重要供应商", "已经成为军队视频指挥领域的重要供应商"],
    793: ["下游行业为各类终端用户"],
    795: ["荣获国家科技进步一等奖"],
    543: ["注册资本"],
    531: ["法定代表人：程家明", "法定代表人:程家明"],
    207: ["15,000.00"],
}
expanded = {}
for item in EVAL["questions"]:
    qid = item["id"]
    pages = set()
    for ev in EVIDENCE.get(qid, []):
        for c in km.chunks:
            if ev in c["text"]:
                pages.add(c["page"])
    expanded[qid] = sorted(pages) or set(item["page"])

CONFIGS = [
    ("rrf_lambda2.5", dict(mode="rrf", lam=2.5)),
    ("rrf_lambda1.0", dict(mode="rrf", lam=1.0)),
    ("sum_0.6_0.4", dict(mode="sum", wd=0.6, ws=0.4)),
    ("sum_0.5_0.5", dict(mode="sum", wd=0.5, ws=0.5)),
    ("sum_0.4_0.6", dict(mode="sum", wd=0.4, ws=0.6)),
]

def evaluate(cfg):
    h1 = h3 = h5 = 0; rr = 0.0; rows = []
    for item in EVAL["questions"]:
        gold = set(expanded[item["id"]])
        hits, conf = engine.retrieve(item["question"], top_k=5, mode=cfg.get("mode"), recall_k=30, **{k:v for k,v in cfg.items() if k!="mode"})
        rank = None
        for i, h in enumerate(hits, 1):
            if h["page"] in gold or (h.get("page_end") in gold if h.get("page_end") else False):
                rank = i; break
        h1 += rank == 1; h3 += bool(rank and rank <= 3); h5 += bool(rank and rank <= 5)
        rr += (1.0 / rank) if rank else 0.0
        rows.append((item["id"], rank))
    n = len(EVAL["questions"])
    return dict(hit1=h1/n, hit3=h3/n, hit5=h5/n, mrr=rr/n, rows=rows)

print("展开后的金标准页码:")
for k, v in expanded.items():
    print("  id%-4s -> %s" % (k, v))
print()
for name, cfg in CONFIGS:
    r = evaluate(cfg)
    print("%-14s Hit@1=%.0f%% Hit@3=%.0f%% Hit@5=%.0f%% MRR=%.4f  %s" % (
        name, 100*r["hit1"], 100*r["hit3"], 100*r["hit5"], r["mrr"],
        " ".join("id%d:%s" % (i, rk) for i, rk in r["rows"])))
