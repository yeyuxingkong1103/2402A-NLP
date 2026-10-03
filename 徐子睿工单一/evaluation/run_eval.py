# -*- coding: utf-8 -*-
# 工单编号：人工智能 NLP-RAG-基于 PDF 文档的问答系统优化
# 模块：run_eval —— 检索评估（10 题金标准）
# 说明：对 evaluation/eval_questions.json 逐题检索，计算 Hit@1/@3/@5、MRR，
#       并输出每题命中的页码与片段，便于人工核对。

import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "app"))
sys.stdout.reconfigure(encoding="utf-8")

import config      # noqa: E402
import engine      # noqa: E402

EVAL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_questions.json")


def expand_gold(item, chunks):
    """金标准页 = 规范出处页 ∪ 所有含证据串的页（避免同一句多页重复被误判 miss）。"""
    gold = set(item.get("page") or [])
    for ev in item.get("evidence") or []:
        for c in chunks:
            if ev in c["text"]:
                gold.add(c["page"])
    return gold


def hit_pages(hits):
    ps = set()
    for h in hits:
        ps.add(h["page"])
        if h.get("page_end"):
            ps.add(h["page_end"])
    return ps


def main():
    data = json.load(open(EVAL_PATH, encoding="utf-8"))
    qs = data["questions"]
    import kb as kb_mod
    chunks = kb_mod.get_kb().chunks
    top_k = config.TOP_K
    res = []
    hit1 = hit3 = hit5 = 0
    rr_sum = 0.0
    for item in qs:
        hits, conf = engine.retrieve(item["question"], top_k=max(top_k, 5))
        gold = expand_gold(item, chunks)
        rank = None
        for i, h in enumerate(hits, 1):
            if h["page"] in gold or (h.get("page_end") in gold if h.get("page_end") else False):
                rank = i
                break
        if rank == 1:
            hit1 += 1
        if rank and rank <= 3:
            hit3 += 1
        if rank and rank <= 5:
            hit5 += 1
        if rank:
            rr_sum += 1.0 / rank
        res.append({"id": item["id"], "question": item["question"], "gold_pages": sorted(gold),
                    "hit_rank": rank, "confidence": round(conf, 3),
                    "top_pages": [h["page"] for h in hits[:5]]})
        mark = "[OK]" if rank else "[MISS]"
        print("%s id%-4s rank=%-4s conf=%.3f gold=%s top=%s" %
              (mark, item["id"], rank, conf, sorted(gold), [h["page"] for h in hits[:5]]))
    n = len(qs)
    print("\n==== 汇总 (n=%d, top_k=%d) ====" % (n, top_k))
    print("Hit@1 = %.1f%% (%d/%d)" % (100 * hit1 / n, hit1, n))
    print("Hit@3 = %.1f%% (%d/%d)" % (100 * hit3 / n, hit3, n))
    print("Hit@5 = %.1f%% (%d/%d)" % (100 * hit5 / n, hit5, n))
    print("MRR   = %.4f" % (rr_sum / n))
    out = os.path.join(os.path.dirname(EVAL_PATH), "retrieval_result.json")
    json.dump({"summary": {"hit@1": hit1 / n, "hit@3": hit3 / n, "hit@5": hit5 / n,
                           "mrr": rr_sum / n, "n": n}, "detail": res},
              open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("saved ->", out)


if __name__ == "__main__":
    main()
