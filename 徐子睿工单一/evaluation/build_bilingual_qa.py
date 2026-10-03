# -*- coding: utf-8 -*-
# 工单编号：人工智能 NLP-RAG-基于 PDF 文档的问答系统优化
# 模块：build_bilingual_qa —— 中英双语问答数据构建
# 说明：对中文 10 题 + 英文 10 题（同 id、同金标准）逐一走 RAG 链路作答，
#       落盘：① qa_bilingual_result.json（含答案 / 检索上下文，供 RAGAS 复用）
#             ② report_bilingual.md（人读对照报告）
import os
import sys
import json
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "app"))
sys.stdout.reconfigure(encoding="utf-8")

import config        # noqa: E402
import engine        # noqa: E402
import kb as kb_mod  # noqa: E402

ZH = json.load(open(os.path.join(HERE, "eval_questions.json"), encoding="utf-8"))["questions"]
EN = json.load(open(os.path.join(HERE, "eval_questions_en.json"), encoding="utf-8"))["questions"]
chunks = kb_mod.get_kb().chunks


def expand_gold(item):
    gold = set(item.get("page") or [])
    for ev in item.get("evidence") or []:
        for c in chunks:
            if ev in c["text"]:
                gold.add(c["page"])
    return sorted(gold)


records, md = [], []
md += ["# 中英双语问答结果对照", "",
       "> 工单编号：人工智能 NLP-RAG-基于 PDF 文档的问答系统优化",
       "> 链路：bge-m3 向量 + jieba BM25 → 加权 RRF → qwen2:7b 生成（温度 0.1）", ""]

for lang, qs in (("zh", ZH), ("en", EN)):
    md += ["", "---", "", "## %s 问题（%s）" % ("中文" if lang == "zh" else "English", lang), ""]
    gold_map = {q["id"]: q for q in ZH}
    for item in qs:
        qid = item["id"]
        q = item["question"]
        t0 = time.time()
        r = engine.answer(q)
        ms = int((time.time() - t0) * 1000)
        gold_item = gold_map[qid]
        pages = expand_gold(gold_item)
        cited = sorted({h["page"] for h in r["hits"]})
        cite_ok = bool(set(cited) & set(pages))
        records.append({
            "id": qid, "lang": lang, "question": q,
            "answer": r["answer"], "refused": r["refused"],
            "confidence": r["confidence"], "cost_ms": ms,
            "contexts": [h["text"] for h in r["hits"]],
            "cited_pages": cited, "gold_pages": pages, "cite_ok": cite_ok,
            "gold_answer": gold_item["gold_answer"],
        })
        md += [
            "### id%s ｜ %s" % (qid, q), "",
            "- 耗时 %d ms ｜ 置信度 %.3f ｜ 拒答 %s ｜ 引用命中金标准页 %s ｜ 命中页 %s"
            % (ms, r["confidence"] or 0, r["refused"], "✅" if cite_ok else "❌", cited),
            "- **金标准**：%s" % gold_item["gold_answer"],
            "", "> " + (r["answer"].strip().replace("\n", "\n> ")), "",
        ]

json.dump({"records": records}, open(os.path.join(HERE, "qa_bilingual_result.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
open(os.path.join(HERE, "report_bilingual.md"), "w", encoding="utf-8").write("\n".join(md))

n = len(records)
zh = [r for r in records if r["lang"] == "zh"]
en = [r for r in records if r["lang"] == "en"]
print("records:", n)
print("zh 引用命中金标准页: %d/%d" % (sum(r["cite_ok"] for r in zh), len(zh)))
print("en 引用命中金标准页: %d/%d" % (sum(r["cite_ok"] for r in en), len(en)))
print("zh 拒答: %d  en 拒答: %d" % (sum(r["refused"] for r in zh), sum(r["refused"] for r in en)))
print("saved -> qa_bilingual_result.json / report_bilingual.md")
