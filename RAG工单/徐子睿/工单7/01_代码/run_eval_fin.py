# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-功能测试及评估任务
# 关联工单：人工智能NLP-RAG-PDF文档的表格解析及检索优化 | 人工智能NLP-RAG-图像内容解析及检索优化 | 人工智能NLP-RAG-Query理解优化任务 | 人工智能NLP-RAG-混合检索任务 | 人工智能NLP-RAG-功能测试及评估任务
# 模块：run_eval_fin —— 金融年报 QA 功能测试与检索评估（工单 07 产出物二）
# 说明：对 10 道金融题逐题做「检索 + 生成」，输出：检索命中页 / 是否命中金标准页 / 关键数值是否命中 /
#       生成答案 / 耗时；落盘 金融问答检索结果.md + fin_qa_result.json。
import os
import sys
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "app"))
sys.stdout.reconfigure(encoding="utf-8")

import config      # noqa: E402
import engine      # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
Q_PATH = os.path.join(HERE, "eval_questions_fin.json")
OUT_MD = os.path.join(HERE, "金融问答检索结果.md")
OUT_JSON = os.path.join(HERE, "fin_qa_result.json")


def main():
    items = json.load(open(Q_PATH, encoding="utf-8"))["questions"]
    import kb as kb_mod
    km = kb_mod.get_kb()

    def gold_pages(it):
        """金标准页 = 人工标注页 ∪ 「含答案数值的页」（多页金标准，可验证）。"""
        g = set(it.get("gold_page") or [])
        kws = it.get("gold_kw") or []
        for c in km.chunks:
            if c["doc"] == it.get("company") and any(k in c["text"] for k in kws):
                g.add(c["page"])
        return g

    n = len(items)
    h1 = h3 = h5 = 0
    rr = 0.0
    kw_ok_n = 0
    rows = []
    for it in items:
        gold = gold_pages(it)
        kws = it.get("gold_kw") or []
        t0 = time.time()
        hits, conf = engine.retrieve(it["question"], top_k=5)
        r = engine.answer(it["question"])
        cost = int((time.time() - t0) * 1000)

        pages = [h["page"] for h in hits]
        rank = next((i for i, h in enumerate(hits, 1)
                     if h["page"] in gold or (h.get("page_end") in gold if h.get("page_end") else False)), None)
        hit_ctx = any(any(k in h["text"] for k in kws) for h in hits) if kws else bool(rank)
        kw_ok = all(any(k in r["answer"] for k in [k]) for k in kws) if kws else bool(rank)
        h1 += rank == 1
        h3 += bool(rank and rank <= 3)
        h5 += bool(rank and rank <= 5)
        rr += (1.0 / rank) if rank else 0.0
        kw_ok_n += kw_ok
        rows.append({"id": it["id"], "company": it["company"], "question": it["question"],
                     "gold_answer": it.get("gold_answer"), "gold_page": sorted(gold),
                     "gold_kw": kws, "top_pages": pages, "contexts": [h["text"] for h in hits],
                     "hit_rank": rank, "hit_ctx": hit_ctx, "kw_ok": kw_ok,
                     "answer": r["answer"], "refused": r.get("refused"),
                     "confidence": r.get("confidence"), "cost_ms": cost})
        print("[%s] id%-3s %s rank=%-4s kw_ok=%s top=%s" %
              ("OK" if rank else "MISS", it["id"], it["company"], rank, kw_ok, pages[:5]))

    summary = {"n": n, "hit@1": h1 / n, "hit@3": h3 / n, "hit@5": h5 / n, "mrr": rr / n,
               "kw_hit_rate": kw_ok_n / n}
    print("\n==== 金融问答检索评估 (n=%d) ====" % n)
    print("Hit@1=%.1f%%  Hit@3=%.1f%%  Hit@5=%.1f%%  MRR=%.4f  关键数值命中率=%.1f%%" %
          (100 * summary["hit@1"], 100 * summary["hit@3"], 100 * summary["hit@5"],
           summary["mrr"], 100 * summary["kw_hit_rate"]))

    json.dump({"summary": summary, "detail": rows}, open(OUT_JSON, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    md = ["# 工单 07 · 金融年报问答检索结果", "",
          "> 工单编号：人工智能NLP-RAG-功能测试及评估任务",
          "> 语料：ccf_competition 年报（9 份）/ 索引 %d 块；指标为「是否命中金标准页」。" % len(__import__("kb").get_kb().chunks),
          "> **Hit@1 %.1f%% ｜ Hit@3 %.1f%% ｜ Hit@5 %.1f%% ｜ MRR %.4f ｜ 关键数值命中率 %.1f%%**" %
          (100 * summary["hit@1"], 100 * summary["hit@3"], 100 * summary["hit@5"],
           summary["mrr"], 100 * summary["kw_hit_rate"]), ""]
    for r in rows:
        md += ["## 第 %d 题 · %s" % (r["id"], r["company"]), "",
               "- **Q**：%s" % r["question"],
               "- **A**：%s" % r["answer"].replace("\n", " "),
               "- 检索到页码：%s ｜ 金标准页：%s ｜ 命中排名：%s ｜ 关键数值命中：%s ｜ 置信度 %s ｜ %d ms" %
               (r["top_pages"], r["gold_page"], r["hit_rank"], r["kw_ok"], r["confidence"], r["cost_ms"]), ""]
    open(OUT_MD, "w", encoding="utf-8").write("\n".join(md))
    print("saved ->", OUT_MD)


if __name__ == "__main__":
    main()
