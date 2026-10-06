# -*- coding: utf-8 -*-
# 工单编号：人工智能 NLP-RAG-基于 PDF 文档的问答系统优化
# 关联工单：人工智能NLP-RAG-PDF文档的表格解析及检索优化 | 人工智能NLP-RAG-图像内容解析及检索优化 | 人工智能NLP-RAG-Query理解优化任务 | 人工智能NLP-RAG-混合检索任务
# 模块：run_eval_en —— 多语言（英文）检索评估
# 说明：与 run_eval.py 同口径，但用英文问句（eval_questions_en.json）；
#       金标准页码/证据沿用中文题库同 id 的标注，用于验证跨语（英→中）检索是否达标。

import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "app"))
sys.stdout.reconfigure(encoding="utf-8")

import config      # noqa: E402
import engine      # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ZH_PATH = os.path.join(HERE, "eval_questions.json")
EN_PATH = os.path.join(HERE, "eval_questions_en.json")


def expand_gold(item, chunks):
    gold = set(item.get("page") or [])
    for ev in item.get("evidence") or []:
        for c in chunks:
            if ev in c["text"]:
                gold.add(c["page"])
    return gold


def main():
    zh = {q["id"]: q for q in json.load(open(ZH_PATH, encoding="utf-8"))["questions"]}
    en = json.load(open(EN_PATH, encoding="utf-8"))["questions"]
    import kb as kb_mod
    chunks = kb_mod.get_kb().chunks
    n = len(en)
    h1 = h3 = h5 = 0
    rr = 0.0
    detail = []
    for item in en:
        gold = expand_gold(zh[item["id"]], chunks)
        hits, conf = engine.retrieve(item["question"], top_k=max(config.TOP_K, 5))
        pages = set()
        for h in hits:
            pages.add(h["page"])
            if h.get("page_end"):
                pages.add(h["page_end"])
        rank = None
        for i, h in enumerate(hits, 1):
            if h["page"] in gold or (h.get("page_end") in gold if h.get("page_end") else False):
                rank = i
                break
        h1 += rank == 1
        h3 += bool(rank and rank <= 3)
        h5 += bool(rank and rank <= 5)
        rr += (1.0 / rank) if rank else 0.0
        detail.append({"id": item["id"], "question": item["question"], "hit_rank": rank,
                       "confidence": round(conf, 3),
                       "top_pages": [h["page"] for h in hits[:5]]})
        print("%s id%-4s rank=%-4s conf=%.3f top=%s" %
              ("[OK]" if rank else "[MISS]", item["id"], rank, conf,
               [h["page"] for h in hits[:5]]))
    print("\n==== 英文检索汇总 (n=%d) ====" % n)
    print("Hit@1 = %.1f%% (%d/%d)" % (100 * h1 / n, h1, n))
    print("Hit@3 = %.1f%% (%d/%d)" % (100 * h3 / n, h3, n))
    print("Hit@5 = %.1f%% (%d/%d)" % (100 * h5 / n, h5, n))
    print("MRR   = %.4f" % (rr / n))
    out = os.path.join(HERE, "retrieval_result_en.json")
    json.dump({"summary": {"hit@1": h1 / n, "hit@3": h3 / n, "hit@5": h5 / n, "mrr": rr / n, "n": n},
               "detail": detail}, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("saved ->", out)


if __name__ == "__main__":
    main()
