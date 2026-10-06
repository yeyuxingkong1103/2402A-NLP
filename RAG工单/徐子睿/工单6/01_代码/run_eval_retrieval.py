# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-混合检索任务
# 关联工单：人工智能NLP-RAG-PDF文档的表格解析及检索优化 | 人工智能NLP-RAG-图像内容解析及检索优化 | 人工智能NLP-RAG-Query理解优化任务 | 人工智能NLP-RAG-混合检索任务
# 模块：run_eval_retrieval —— 检索策略 × 重排器 对比评估
# 说明：在 10 题金标准上对比「向量 / 全文 / 混合 / RRF」四种策略与
#       「无 / TF-IDF / LLM / 用户反馈」四种重排器，输出
#       召回率@k（页面级）、准确率@k（页面级）、题级命中率@k、MRR。
# 用法：python evaluation/run_eval_retrieval.py [--strategies vector,fulltext,hybrid,rrf]
#            [--rerankers none,tfidf,feedback,llm] [--k 5] [--pool 20]
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "app"))
sys.stdout.reconfigure(encoding="utf-8")

import kb as kb_mod      # noqa: E402
import retrieval         # noqa: E402

ZH = os.path.join(HERE, "eval_questions.json")
PDF2 = os.path.join(HERE, "eval_questions_pdf2.json")


def load_questions():
    qs = []
    for path in (ZH, PDF2):
        if not os.path.exists(path):
            continue
        for q in json.load(open(path, encoding="utf-8"))["questions"]:
            gold = q.get("gold_page") or q.get("page") or []
            if gold:
                qs.append({"id": q["id"], "question": q["question"], "gold": set(gold)})
    return qs


def expand_gold(q, chunks):
    """允许用证据串扩展金标准页（与既有评估同口径）。"""
    return set(q["gold"])


def evaluate(qs, chunks, strategy, reranker, k=5, pool=20):
    n = len(qs)
    rec_sum = prec_sum = hit = 0.0
    rr = 0.0
    rows = []
    for q in qs:
        hits = retrieval.search(q["question"], strategy=strategy, reranker=reranker,
                                top_k=k, rerank_pool=max(pool, k))
        got = [h["page"] for h in hits]
        got_set = set(got)
        gold = q["gold"]
        inter = got_set & gold
        rec = len(inter) / len(gold) if gold else 0.0
        prec = len(inter) / len(got) if got else 0.0
        rank = next((i for i, p in enumerate(got, 1) if p in gold), None)
        rec_sum += rec
        prec_sum += prec
        hit += bool(rank)
        rr += (1.0 / rank) if rank else 0.0
        rows.append({"id": q["id"], "rank": rank, "recall": round(rec, 3),
                     "precision": round(prec, 3), "pages": got})
    return {"strategy": strategy, "reranker": reranker, "n": n,
            "recall@k": round(rec_sum / n, 4), "precision@k": round(prec_sum / n, 4),
            "hit@k": round(hit / n, 4), "mrr": round(rr / n, 4)}, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategies", default="vector,fulltext,hybrid,rrf")
    ap.add_argument("--rerankers", default="none,tfidf,feedback")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--pool", type=int, default=20)
    a = ap.parse_args()

    chunks = kb_mod.get_kb().chunks
    qs = load_questions()
    strategies = [s.strip() for s in a.strategies.split(",") if s.strip()]
    rerankers = [r.strip() for r in a.rerankers.split(",") if r.strip()]

    md = ["# 检索策略 × 重排器 对比评估", "",
          "> 工单编号：人工智能NLP-RAG-混合检索任务",
          "> 题目：中文 10 题 + 招股说明书2（力源信息）id1-4；指标为页面级。",
          "> 召回率@k = 平均(命中金标准页数/金标准页总数)；准确率@k = 平均(命中金标准页数/返回页数)。", "",
          "| 策略 | 重排器 | 召回率@%d | 准确率@%d | 题级命中率@%d | MRR |" % (a.k, a.k, a.k),
          "|---|---|---|---|---|---|"]
    all_res = []
    for st in strategies:
        for rk in rerankers:
            res, rows = evaluate(qs, chunks, st, rk, k=a.k, pool=a.pool)
            all_res.append({"summary": res, "detail": rows})
            md.append("| %s | %s | %.1f%% | %.1f%% | %.1f%% | %.4f |" % (
                st, rk, 100 * res["recall@k"], 100 * res["precision@k"], 100 * res["hit@k"], res["mrr"]))
            print("%-9s %-9s 召回=%.1f%% 准确=%.1f%% 命中=%.1f%% MRR=%.4f" % (
                st, rk, 100 * res["recall@k"], 100 * res["precision@k"], 100 * res["hit@k"], res["mrr"]))

    json.dump(all_res, open(os.path.join(HERE, "retrieval_strategies.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    open(os.path.join(HERE, "检索策略对比.md"), "w", encoding="utf-8").write("\n".join(md))
    print("saved -> 检索策略对比.md / retrieval_strategies.json")


if __name__ == "__main__":
    main()
