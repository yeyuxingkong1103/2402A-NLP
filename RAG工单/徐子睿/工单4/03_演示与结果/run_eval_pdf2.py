# -*- coding: utf-8 -*-
# 工单编号：人工智能 NLP-RAG-PDF文档的表格解析及检索优化（03）
#          人工智能 NLP-RAG-图像内容解析及检索优化（04）
# 模块：run_eval_pdf2 —— 《招股说明书2.pdf》新题库检索评估（表格 03 + 图像 04）
# 说明：对 eval_questions_pdf2.json 的 6 道题逐题检索，判定 top_k 内是否命中金标准页；
#       图像题（id5/id6）单独统计，便于与工单 04 的“图像语义解析”效果对齐。

import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "app"))
sys.stdout.reconfigure(encoding="utf-8")

import config      # noqa: E402
import engine      # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
Q_PATH = os.path.join(HERE, "eval_questions_pdf2.json")
OUT = os.path.join(HERE, "retrieval_result_pdf2.json")


def main():
    items = json.load(open(Q_PATH, encoding="utf-8"))["questions"]
    n = len(items)
    h1 = h3 = h5 = 0
    rr = 0.0
    detail = []
    for it in items:
        gold = set(it.get("gold_page") or [])
        hits, conf = engine.retrieve(it["question"], top_k=max(config.TOP_K, 5))
        rank = None
        for i, h in enumerate(hits, 1):
            pages = {h.get("page")}
            if h.get("page_end"):
                pages.add(h.get("page_end"))
            if pages & gold:
                rank = i
                break
        h1 += rank == 1
        h3 += bool(rank and rank <= 3)
        h5 += bool(rank and rank <= 5)
        rr += (1.0 / rank) if rank else 0.0
        top = [(h.get("page"), h.get("type")) for h in hits[:5]]
        detail.append({"id": it["id"], "task": it.get("task"), "question": it["question"],
                       "gold_page": sorted(gold), "hit_rank": rank,
                       "confidence": round(conf, 3), "top": top})
        print("%s id%-3s [task %s] rank=%-5s conf=%.3f gold=%s top=%s" %
              ("[OK]  " if rank else "[MISS]", it["id"], it.get("task"), rank, conf, sorted(gold), top))
    print("\n==== 招股说明书2 检索汇总 (n=%d) ====" % n)
    print("Hit@1 = %.1f%% (%d/%d)" % (100 * h1 / n, h1, n))
    print("Hit@3 = %.1f%% (%d/%d)" % (100 * h3 / n, h3, n))
    print("Hit@5 = %.1f%% (%d/%d)" % (100 * h5 / n, h5, n))
    print("MRR   = %.4f" % (rr / n))
    json.dump({"summary": {"hit@1": h1 / n, "hit@3": h3 / n, "hit@5": h5 / n, "mrr": rr / n, "n": n},
               "detail": detail}, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("saved ->", OUT)


if __name__ == "__main__":
    main()
