# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-基于Graph RAG实现金融问答
# 关联工单：人工智能NLP-RAG-基于Graph RAG实现金融问答 | 人工智能NLP-RAG-Graph RAG优化任务
# 模块：run_eval_graph —— Graph RAG 检索评估（工单 08 产出物二）
# 说明：对题集逐题执行 Graph RAG 检索，输出「检索结果 + 解析出的知识图谱结构（三元组/子图）」，
#       并与工单 07 的常规 RAG 结果对比（命中率、MRR）。同时生成「检索结果.md」。
import os
import sys
import json
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "app"))
sys.stdout.reconfigure(encoding="utf-8")

import engine       # noqa: E402
import kb as kb_mod  # noqa: E402
import graph_rag    # noqa: E402

Q_PATH = os.path.join(HERE, "eval_questions_fin.json")
OUT_MD = os.path.join(HERE, "GraphRAG检索结果.md")
OUT_JSON = os.path.join(HERE, "graph_rag_result.json")


def _compress(text, keys, win=90, cap=420):
    """工单 09 · 上下文压缩（实现见 app/ctxopt.py，评测与线上走同一份逻辑）。"""
    from ctxopt import compress_context
    return compress_context(text, keys, win=win, cap=cap)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--topk", type=int, default=5, help="送入生成/评估的上下文条数（工单 09 上下文裁剪）")
    ap.add_argument("--out", default="graph_qa_result.json")
    ap.add_argument("--compress", action="store_true", help="工单 09：上下文压缩（只留证据窗口）")
    a = ap.parse_args()
    TOPK = a.topk
    km = kb_mod.get_kb()
    items = json.load(open(Q_PATH, encoding="utf-8"))["questions"]

    def gold_pages(it):
        g = set(it.get("gold_page") or [])
        for c in km.chunks:
            if c["doc"] == it.get("company") and any(k in c["text"] for k in (it.get("gold_kw") or [])):
                g.add(c["page"])
        return g

    rows = []
    for mode in ("baseline", "graph"):
        h1 = h3 = h5 = 0
        rr = 0.0
        for it in items:
            g = gold_pages(it)
            t0 = time.time()
            if mode == "graph":
                hits, conf, info = graph_rag.retrieve(it["question"], top_k=TOPK)
            else:
                hits, conf = engine.retrieve(it["question"], top_k=5)
                info = None
            cost = int((time.time() - t0) * 1000)
            p = [h["page"] for h in hits]
            rank = next((i for i, x in enumerate(hits, 1) if x["page"] in g), None)
            h1 += rank == 1
            h3 += bool(rank and rank <= 3)
            h5 += bool(rank and rank <= 5)
            rr += (1.0 / rank) if rank else 0.0
            if mode == "graph":
                import engine as _eng
                import prompts as _pr
                import llm as _llm
                qu = _eng.query_understanding(it["question"])
                gctx = [h["text"] for h in hits]
                hits_llm = hits
                if a.compress:
                    from ctxopt import compress_hits
                    hits_llm = compress_hits(hits, info["entities"])
                    gctx = [h["text"] for h in hits_llm]
                txt = ""
                if hits and conf >= _eng.REFUSE_SCORE:
                    ctx = _pr.build_context(hits_llm)
                    ctx = ctx[: _eng.MAX_CONTEXT_CHARS]
                    txt = _eng.clean_answer(_llm.chat(_pr.rag_messages(it["question"], hits_llm, lang=qu["lang"]), temperature=0.1))
                rows.append({"id": it["id"], "company": it["company"], "question": it["question"],
                             "gold_answer": it.get("gold_answer"),
                             "top_pages": p, "gold_pages": sorted(g), "hit_rank": rank,
                             "graph_triples": info["graph_triples"], "entities": info["entities"],
                             "sources": [h.get("source") for h in hits], "contexts": gctx,
                             "answer": txt, "cost_ms": cost})
            print("[%s %s] id%-3s rank=%-4s top=%s" % (mode, "OK" if rank else "MISS", it["id"], rank, p[:5]))
        n = len(items)
        print("== %s: Hit@1=%.1f%% Hit@3=%.1f%% Hit@5=%.1f%% MRR=%.4f" %
              (mode, 100 * h1 / n, 100 * h3 / n, 100 * h5 / n, rr / n))
        if mode == "graph":
            summary = {"n": n, "hit@1": h1 / n, "hit@3": h3 / n, "hit@5": h5 / n, "mrr": rr / n}

    json.dump({"summary": summary, "detail": rows}, open(OUT_JSON, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    # RAGAS 用：Graph RAG 的 答案 + 上下文（工单 09 前后对比）
    json.dump({"summary": summary, "detail": rows}, open(os.path.join(HERE, a.out), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    if a.out == "graph_qa_result.json":
        pass

    md = ["# 工单 08 · Graph RAG 金融问答检索结果", "",
          "> 工单编号：人工智能NLP-RAG-基于Graph RAG实现金融问答",
          "> 检索方式：实体链接（公司/指标/年份）→ 知识图谱三元组定位来源页 → 不足 top-5 时用混合检索补足。",
          "> **Graph RAG：Hit@1 %.1f%% ｜ Hit@3 %.1f%% ｜ Hit@5 %.1f%% ｜ MRR %.4f**" %
          (100 * summary["hit@1"], 100 * summary["hit@3"], 100 * summary["hit@5"], summary["mrr"]), ""]
    for r in rows:
        md += ["## 第 %s 题 · %s" % (r["id"], r["company"]), "",
               "- **Q**：%s" % r["question"],
               "- **识别实体**：%s" % r["entities"],
               "- **知识图谱结构（公司–指标–值@年份–页）**："]
        for t in r["graph_triples"][:6]:
            md.append("    - %s —[披露指标]→ %s = %s%s（%s年，p%s）" %
                      (t["公司"], t["指标"], t["值"], t["单位"], t["年份"], t["页"]))
        md += ["- **检索到页码**：%s ｜ 金标准页：%s ｜ 命中排名：%s ｜ 来源：%s ｜ %d ms" %
               (r["top_pages"], r["gold_pages"], r["hit_rank"], r["sources"], r["cost_ms"]), ""]
    open(OUT_MD, "w", encoding="utf-8").write("\n".join(md))
    print("saved ->", OUT_MD)


if __name__ == "__main__":
    main()
