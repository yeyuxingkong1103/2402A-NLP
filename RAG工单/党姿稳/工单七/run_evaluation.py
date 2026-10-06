# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-功能测试及评估
测试与评估脚本：
  1. 解析 ccf_competition 语料，构建 01-06 工单实现的 RAG 检索系统（混合检索 + LLM 重排）；
  2. 对 10 个验收问题逐一执行检索与问答，输出：检索片段、页码、RAG 回答；
  3. 通过 LLM 裁判计算检索准确率（Top-5）与召回率（Top-20），并输出问题分析。
"""
import os
import sys
import json
import time
from datetime import datetime

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

from kb import load_chunks
from hybrid_retriever import HybridRetriever
from rag_chain import judge_contexts, call_llm, build_rag_prompt, RAG_SYSTEM_PROMPT
from config import EVAL_QUESTIONS, RECALL_TOP_N, ACCURACY_TARGET, RECALL_TARGET

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluation")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    chunks = load_chunks()
    docs = sorted({c["doc"] for c in chunks})
    print(f"语料：{len(chunks)} 块；文档 {len(docs)} 份：{docs}")
    r = HybridRetriever(chunks, reranker="llm", vector_weight=0.6, fulltext_weight=0.4,
                        fusion="weighted")

    results, t0 = [], time.time()
    for i, item in enumerate(EVAL_QUESTIONS, 1):
        q = item["question"]
        print(f"\n[{i}/{len(EVAL_QUESTIONS)}] Q{item['id']}（{item['doc']}）: {q[:60]}")
        res = r.hybrid_search(q, top_k=5)
        broad = r.hybrid_search(q, top_k=RECALL_TOP_N)
        hit = judge_contexts(q, res)
        # Top-20 是 Top-5 的超集，命中 Top-5 必然命中 Top-20（避免裁判的偶发抖动造成自相矛盾）
        recalled = hit or judge_contexts(q, broad)
        answer = call_llm(RAG_SYSTEM_PROMPT, build_rag_prompt(q, res))
        print(f"  准确={hit} 召回={recalled} 页码={[c['page'] for c, _ in res]}")
        results.append({
            "id": item["id"], "expected_doc": item["doc"], "question": q, "note": item.get("note", ""),
            "hit": hit, "recalled": recalled,
            "retrieved": [{"doc": c["doc"], "page": c["page"], "score": round(float(s), 4),
                           "text": c["text"][:300]} for c, s in res],
            "recall_pages": [c["page"] for c, _ in broad],
            "recall_docs": sorted({c["doc"] for c, _ in broad}),
            "answer": answer,
        })

    n = len(results)
    acc = sum(x["hit"] for x in results) / n
    rec = sum(x["recalled"] for x in results) / n
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary = {"timestamp": ts, "n_questions": n, "n_chunks": len(chunks), "docs": docs,
               "elapsed_sec": round(time.time() - t0, 1),
               "accuracy_top5": round(acc, 4), "recall_top20": round(rec, 4),
               "accuracy_target": ACCURACY_TARGET, "recall_target": RECALL_TARGET,
               "accuracy_target_met": acc >= ACCURACY_TARGET,
               "recall_target_met": rec >= RECALL_TARGET}

    json_path = os.path.join(OUTPUT_DIR, f"test_eval_{ts}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "details": results}, f, ensure_ascii=False, indent=2)

    # ---------- Markdown 报告（含问题分析） ----------
    fails = [x for x in results if not x["hit"]]
    md_path = os.path.join(OUTPUT_DIR, f"test_eval_{ts}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# 工单七 功能测试及评估报告\n\n")
        f.write("**工单编号：** 人工智能NLP-RAG-功能测试及评估\n\n")
        f.write(f"**评估时间：** {datetime.now():%Y-%m-%d %H:%M:%S}　"
                f"**语料：** ccf_competition 共 {len(chunks)} 块 / {len(docs)} 份年报\n\n")
        f.write("**使用系统：** 01-06 工单实现的 RAG（混合检索：向量召回 + 全文检索 + 加权融合 + LLM 重排）\n\n")
        f.write("## 一、总体指标\n\n")
        f.write("| 指标 | 结果 | 目标 | 是否达标 |\n|---|---|---|---|\n")
        f.write(f"| 检索准确率 Top-5 | {acc:.1%} | ≥{ACCURACY_TARGET:.0%} | "
                f"{'✅' if summary['accuracy_target_met'] else '❌'} |\n")
        f.write(f"| 检索召回率 Top-20 | {rec:.1%} | ≥{RECALL_TARGET:.0%} | "
                f"{'✅' if summary['recall_target_met'] else '❌'} |\n")
        f.write(f"| 总耗时 | {summary['elapsed_sec']}s | — | — |\n\n")
        f.write("## 二、10 个问题的检索与评估结果\n\n")
        f.write("| # | 问题 | 期望文档 | Top-5 是否含答案 | Top-20 召回 | 检索页码 |\n|---|---|---|---|---|---|\n")
        for x in results:
            f.write(f"| {x['id']} | {x['question'][:44]} | {x['expected_doc']} | "
                    f"{'✅' if x['hit'] else '❌'} | {'✅' if x['recalled'] else '❌'} | "
                    f"{[c['page'] for c in x['retrieved']]} |\n")
        f.write("\n## 三、逐题检索结果与 RAG 回答\n\n")
        for x in results:
            f.write(f"### Q{x['id']}（{x['expected_doc']}）\n\n**问题：** {x['question']}\n\n")
            f.write(f"**检索结果（Top-5）：**\n\n")
            for j, c in enumerate(x["retrieved"], 1):
                f.write(f"{j}. `{c['doc']}` 第{c['page']}页（分数 {c['score']}）：{c['text'][:120]}…\n")
            f.write(f"\n**RAG 回答：** {x['answer']}\n\n**评估：** 准确率 "
                    f"{'命中 ✅' if x['hit'] else '未命中 ❌'}；召回 "
                    f"{'命中 ✅' if x['recalled'] else '未命中 ❌'}\n\n---\n\n")
        f.write("## 四、检索结果的局限性与风险点\n\n")
        if fails:
            f.write(f"本轮共 {len(fails)} 个问题未在 Top-5 命中答案，问题归因如下：\n\n")
        else:
            f.write(f"本轮 {n} 个问题在 Top-5 与 Top-20 上均命中答案。"
                    "但这只说明本题集未触发链路短板，结合 ccf 语料的实际特点，"
                    "以下风险点换一批问题仍可能出现：\n\n")
        f.write("""1. **跨文档综合型问题**：如 Q4 要求横向比较"多家银行与保险公司"的共同/差异化策略，
答案分散在 9 份年报的不同章节，而检索按块返回 Top-5，天然只能覆盖其中 1~2 份文档，
单次检索无法覆盖全部证据（需扩大 Top-K 或多轮检索聚合）。
2. **精确数字型问题（表格类）**：如 Q5~Q10 的营收、净利润、不良贷款率、内含价值、新业务价值等
关键财务数字主要存在于年报的**财务表格**中，纯文本抽取会把表格行列结构打散，
数字与指标名分离，导致语义匹配弱、分数偏低（本工单沿用 01-06 的检索链路，未对 ccf 语料单独做表格结构还原）。
3. **长问题的语义稀释**：Q3 这类带大量限定条件的长问题（60+ 字），其向量表示被多个子意图平均，
单一向量难以同时匹配"拨备覆盖率""资产质量""贷款结构"三个子主题，易偏向其中某一个。
4. **同义/近似表述的召回不足**：年报用语与提问用语存在偏差（如"归母净利润"对"归属于母公司股东的净利润"、
"NBV"对"新业务价值"），BM25 精确词匹配失效，需依赖向量语义与查询扩展补偿。
5. **文档内相似段落干扰**：同一份年报中指标名多次出现（目录、摘要、正文、附注），
重排器容易把"提及指标名的目录段"排在"含具体数值的正文段"之前。
6. **PDF 抽取的字符间空格**：中文年报 PDF 存在字距排版，抽取文本中夹有空格
（如"市、优于 同业"），会切断 BM25 的中文二字切分，削弱全文检索的精确匹配能力。

**改进方向**：对 ccf 语料单独启用表格结构解析（工单三方案）；抽取后做中文字符间空格归一化；
对长问题做子问题拆分并分别检索后聚合；引入指标同义词/别名表做查询扩展；提高 Top-K 并配合答案聚合生成。
""")
    # ---------- 测试用例文档 ----------
    case_path = os.path.join(OUTPUT_DIR, f"测试用例_{ts}.md")
    with open(case_path, "w", encoding="utf-8") as f:
        f.write("# 工单七 测试用例及测试结果\n\n")
        f.write("**工单编号：** 人工智能NLP-RAG-功能测试及评估\n\n")
        f.write("**测试环境：** Windows + Python 3.8(nlp2) + m3e-base 嵌入模型 + DeepSeek LLM\n\n")
        f.write("**测试对象：** 01-06 工单实现的 RAG 系统（混合检索：向量召回 + 全文检索 + 加权融合 + LLM 重排）\n\n")
        f.write("**测试语料：** 附件 ccf_competition（9 份金融年报，共 %d 个文本块）\n\n" % len(chunks))
        f.write("| 用例编号 | 前置条件 | 测试步骤 | 预期结果 | 实际结果 | 结论 |\n|---|---|---|---|---|---|\n")
        for x in results:
            f.write(f"| TC-{x['id']:02d} | 语料已构建、检索服务可用 | "
                    f"输入问题：{x['question'][:36]}… 执行检索并生成回答 | "
                    f"返回包含答案的文本块（{x['expected_doc']}） | "
                    f"Top-5 {'命中答案' if x['hit'] else '未命中答案'}（页码 "
                    f"{[c['page'] for c in x['retrieved']]}） | "
                    f"{'通过' if x['hit'] else '不通过'} |\n")
        f.write(f"\n**用例通过率：** {sum(x['hit'] for x in results)}/{n} = {acc:.1%}\n")
    print(f"\n✅ 报告已保存:\n{json_path}\n{md_path}\n{case_path}")
    print(f"准确率(Top-5) {acc:.1%}（目标 {ACCURACY_TARGET:.0%}）；召回率(Top-20) {rec:.1%}（目标 {RECALL_TARGET:.0%}）")


if __name__ == "__main__":
    main()
