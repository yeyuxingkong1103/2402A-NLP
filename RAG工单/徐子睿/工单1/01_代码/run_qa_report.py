# -*- coding: utf-8 -*-
# 工单编号：人工智能 NLP-RAG-基于 PDF 文档的问答系统优化
# 关联工单：人工智能NLP-RAG-PDF文档的表格解析及检索优化 | 人工智能NLP-RAG-图像内容解析及检索优化 | 人工智能NLP-RAG-Query理解优化任务 | 人工智能NLP-RAG-混合检索任务
# 模块：run_qa_report —— 10 题作答 + RAG vs 纯 LLM 对比报告
# 说明：对 10 道评测题分别走 RAG 链路与纯 LLM 链路作答，落盘对照报告（Markdown）。
import os, sys, json, time
sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "app"))
import config, engine, kb as kb_mod
EVAL = json.load(open(os.path.join(HERE, "eval_questions.json"), encoding="utf-8"))
OUT = os.path.join(HERE, "report_rag_vs_llm.md")
chunks = kb_mod.get_kb().chunks


def expand_gold(item):
    gold = set(item.get("page") or [])
    for ev in item.get("evidence") or []:
        for c in chunks:
            if ev in c["text"]:
                gold.add(c["page"])
    return gold


lines = ["# 10 题作答与 RAG vs 纯 LLM 对比报告", "",
         "> 工单编号：人工智能 NLP-RAG-基于 PDF 文档的问答系统优化",
         "> 模型：bge-m3（嵌入）+ qwen2:7b（生成，温度 0.1）；RAG 与纯 LLM 使用同一生成模型。", ""]
summary = []
for item in EVAL["questions"]:
    q = item["question"]
    t0 = time.time()
    r = engine.answer(q)
    rag_ms = int((time.time() - t0) * 1000)
    t1 = time.time()
    p = engine.answer_plain_llm(q)
    llm_ms = int((time.time() - t1) * 1000)
    gold = sorted(expand_gold(item))
    cite_pages = sorted({h["page"] for h in r["hits"]})
    cite_ok = bool(set(cite_pages) & set(gold))
    summary.append((item["id"], rag_ms, llm_ms, r["refused"], cite_ok))
    lines += [
        "## id%s：%s" % (item["id"], q), "",
        "**金标准**：%s" % item["gold_answer"], "",
        "**金标准页码**：%s" % gold, "",
        "### 🔍 RAG（检索增强）",
        "- 耗时：%d ms ｜ 置信度：%.3f ｜ 拒答：%s ｜ 引用命中金标准页：%s" % (
            rag_ms, r["confidence"] or 0, r["refused"], "是" if cite_ok else "否"),
        "- 命中页码：%s" % cite_pages, "",
        "```", r["answer"].strip(), "```", "",
        "### 🧠 纯 LLM（不检索）",
        "- 耗时：%d ms" % llm_ms, "",
        "```", p["answer"].strip(), "```", "",
        "---", "",
    ]

lines.insert(4, "| id | RAG 耗时 | 纯LLM 耗时 | RAG 拒答 | RAG 引用命中 |")
lines.insert(5, "|---|---|---|---|---|")
for i, (qid, rm, lm, rf, ok) in enumerate(summary):
    lines.insert(6 + i, "| %s | %d ms | %d ms | %s | %s |" % (qid, rm, lm, "是" if rf else "否", "✅" if ok else "❌"))

open(OUT, "w", encoding="utf-8").write("\n".join(lines))
print("written ->", OUT)
for qid, rm, lm, rf, ok in summary:
    print("id%-4s RAG %5dms  LLM %5dms  refused=%-5s cite_ok=%s" % (qid, rm, lm, rf, ok))
print("RAG 平均耗时 %.0f ms | 纯LLM 平均耗时 %.0f ms" % (
    sum(s[1] for s in summary) / len(summary), sum(s[2] for s in summary) / len(summary)))
print("引用命中金标准: %d/%d" % (sum(1 for s in summary if s[4]), len(summary)))
