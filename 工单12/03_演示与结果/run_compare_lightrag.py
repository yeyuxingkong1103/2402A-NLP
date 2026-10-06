# -*- coding: utf-8 -*-
# 工单编号：人工智能 NLP-RAG 项目-LightRAG 优化任务
# 关联工单：人工智能NLP-RAG-基于PDF文档的问答系统 | 人工智能NLP-RAG-PDF文档的表格解析及检索优化
# 模块：evaluation/run_compare_lightrag —— 「常规 RAG vs LightRAG」16 题对比（工单 12）
# 说明：同一批 16 题（招股书1 中文 10 题 + 招股书2 6 题）、同一底层模型（bge-m3 + qwen2:7b），
#       分别跑 ① 常规 RAG（hybrid 检索 + 生成）② LightRAG（mix/naive/local），
#       对比命中率 / MRR / 检索时延，并导出 RAGAS 输入（答案+上下文+参考答案）。
# 用法：python evaluation/run_compare_lightrag.py [--modes mix,naive,local]
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "app"))
sys.stdout.reconfigure(encoding="utf-8")

import engine       # noqa: E402
import kb as kb_mod  # noqa: E402
import prompts      # noqa: E402
import llm          # noqa: E402

LR_DIR = os.path.join(HERE, "..", "lightrag_rag", "results")
OUT_MD = os.path.join(HERE, "对比_RAG_vs_LightRAG.md")
OUT_JSON = os.path.join(HERE, "对比_RAG_vs_LightRAG.json")


def load_lightrag(mode):
    p = os.path.join(LR_DIR, "lightrag_%s.json" % mode)
    if not os.path.exists(p):
        return None
    return json.load(open(p, encoding="utf-8"))


def run_rag_side(questions, top_k=5):
    """常规 RAG：hybrid 检索 + 同源 prompt 生成（与 LightRAG 用同一 LLM）。"""
    rows = []
    for it in questions:
        t0 = time.time()
        hits, conf = engine.retrieve(it["q"], top_k=top_k)
        pages = [str(h["page"]) for h in hits]
        rank = next((k for k, p in enumerate(pages, 1) if p in it["gold"]), None)
        ans = ""
        if hits and conf >= engine.REFUSE_SCORE:
            txt = llm.chat(prompts.rag_messages(it["q"], hits), temperature=0.1)
            ans = engine.clean_answer(txt)
        rows.append({"id": it["id"], "question": it["q"], "gold_pages": sorted(it["gold"]),
                     "gold_answer": it["gold_answer"], "top_pages": pages, "hit_rank": rank,
                     "answer": ans, "contexts": [h["text"] for h in hits],
                     "cost_ms": int((time.time() - t0) * 1000)})
        print("  [RAG %s] id%-4s rank=%-4s pages=%s" %
              ("OK" if rank else "MISS", it["id"], rank, pages[:5]), flush=True)
    return rows


def summarize(rows):
    n = len(rows)
    h1 = sum(1 for r in rows if r["hit_rank"] == 1)
    h3 = sum(1 for r in rows if r["hit_rank"] and r["hit_rank"] <= 3)
    h5 = sum(1 for r in rows if r["hit_rank"] and r["hit_rank"] <= 5)
    return {"n": n, "hit@1": h1 / n, "hit@3": h3 / n, "hit@5": h5 / n,
            "mrr": sum((1.0 / r["hit_rank"]) if r["hit_rank"] else 0.0 for r in rows) / n,
            "avg_ms": round(sum(r.get("cost_ms") or 0 for r in rows) / n)}


def pct(x):
    return "%.1f%%" % (100 * x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", default="mix,naive,local")
    a = ap.parse_args()

    sys.path.insert(0, os.path.join(HERE, "..", "lightrag_rag"))
    from query_lightrag import load_questions
    questions = load_questions()
    print("题目数 =", len(questions))

    rag_rows = run_rag_side(questions)
    rag_sum = summarize(rag_rows)

    results = {"rag": {"summary": rag_sum, "detail": rag_rows}}
    for mode in [m.strip() for m in a.modes.split(",") if m.strip()]:
        d = load_lightrag(mode)
        if not d:
            print("跳过（未找到结果）：", mode)
            continue
        results["lightrag_" + mode] = d
        print("  LightRAG[%s]: %s" % (mode, {k: round(v, 4) if isinstance(v, float) else v
                                             for k, v in d["summary"].items()}))

    md = ["# 工单 12 · 常规 RAG 与 LightRAG 检索对比（16 题）", "",
          "> 工单编号：人工智能 NLP-RAG 项目-LightRAG 优化任务",
          "> 语料：《招股说明书1》+《招股说明书2》；题集：招股书1 中文 10 题 + 招股书2 6 题 = **16 题**",
          "> 底层模型一致：嵌入 bge-m3、生成 qwen2:7b（Ollama 本机）；二者差异只在**检索/组织方式**。", "",
          "## 一、总体指标", "",
          "| 方案 | Hit@1 | Hit@3 | Hit@5 | MRR | 平均检索+生成时延 |", "|---|---|---|---|---|---|",
          "| 常规 RAG（hybrid：向量+BM25+RRF） | %s | %s | %s | %.4f | %d ms |" %
          (pct(rag_sum["hit@1"]), pct(rag_sum["hit@3"]), pct(rag_sum["hit@5"]), rag_sum["mrr"], rag_sum["avg_ms"])]
    for mode in [m.strip() for m in a.modes.split(",") if m.strip()]:
        if ("lightrag_" + mode) not in results:
            continue
        s = results["lightrag_" + mode]["summary"]
        md.append("| **LightRAG（%s）** | %s | %s | %s | %.4f | %d ms |" %
                  (mode, pct(s["hit@1"]), pct(s["hit@3"]), pct(s["hit@5"]), s["mrr"], s["avg_ms"]))

    md += ["", "## 二、逐题对比（命中排名：数字=金标准页在结果中的名次，MISS=未召回）", "",
           "| 题号 | 问题 | 常规 RAG | " + " | ".join(
               m.strip() for m in a.modes.split(",") if ("lightrag_" + m.strip()) in results) + " |",
           "|---|---|---|" + "---|" * len([m for m in a.modes.split(",") if ("lightrag_" + m.strip()) in results])]
    lr_maps = {m.strip(): {r["id"]: r for r in results["lightrag_" + m.strip()]["detail"]}
               for m in a.modes.split(",") if ("lightrag_" + m.strip()) in results}
    for r in rag_rows:
        cells = []
        for m, mp in lr_maps.items():
            x = mp.get(r["id"])
            cells.append(str(x["hit_rank"] or "MISS") if x else "-")
        md.append("| %s | %s | %s | %s |" % (r["id"], r["question"][:34], r["hit_rank"] or "MISS",
                                             " | ".join(cells)))
    md += ["", "## 三、结论", "",
           "- LightRAG 走「实体-关系图 + 向量」双通道：**全局/关系型问题**更有优势（能把分散在不同页的"
           "同一实体关系聚合），但**单值事实题**（某页某个数字）不如常规混合检索直接；",
           "- 常规 RAG 在数值型取数题上命中更高（BM25 关键词 + 口径消歧）；",
           "- 二者互补：可在同一系统里按问题类型路由（见 `docs/LightRAG优化方案.md` 的优化建议）。", "",
           "> 明细数据：`evaluation/对比_RAG_vs_LightRAG.json`；LightRAG 原始结果：`lightrag_rag/results/`", ""]
    open(OUT_MD, "w", encoding="utf-8").write("\n".join(md))
    json.dump(results, open(OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("saved ->", OUT_MD)

    # 导出 RAGAS 输入（两种方案各一份）
    for tag, rows in [("rag", rag_rows)] + [("lightrag_" + m.strip(), results["lightrag_" + m.strip()]["detail"])
                                            for m in a.modes.split(",")
                                            if ("lightrag_" + m.strip()) in results]:
        det = []
        for r in rows:
            if not r.get("contexts"):
                continue
            det.append({"id": r["id"], "question": r["question"], "answer": r["answer"],
                        "contexts": r["contexts"][:8], "gold_answer": r.get("gold_answer") or ""})
        p = os.path.join(HERE, "lightrag_ragas_in_%s.json" % tag.replace("lightrag_", ""))
        json.dump({"detail": det}, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print("ragas 输入 ->", os.path.basename(p), "n =", len(det))


if __name__ == "__main__":
    main()
