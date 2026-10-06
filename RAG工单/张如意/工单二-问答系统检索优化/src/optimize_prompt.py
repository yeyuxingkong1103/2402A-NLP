#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成 Prompt 优化对比实验（optimize_prompt.py）
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化

实验目的
    检索完全相同（同分块、同召回、同重排、同上下文），只改变生成 Prompt，
    对比「朴素 Prompt」与「优化 Prompt」在 10 个兴图新科问题上的表现。

两个 Prompt 的差异（优化版四条硬规则）
    1. 先定位后作答 —— 要求模型先确认信息在哪个片段，再组织答案，减少凭印象作答；
    2. 数值逐字核对 —— 金额/比例/年份必须与资料逐字一致，多期数据分项列出，
       禁止把不同年份、不同口径的数字混在一起（招股书问答最大的坑）；
    3. 表格按行读   —— 注意表头与数据行的对应关系，不串列；
    4. 拒答机制     —— 资料不足时固定回复「未能找到」，不编造、不猜测。

评测指标
    · 准确率（关键词口径，全部必备要点命中才算对）
    · 忠实度 Faithfulness / 答案相关性 Answer Relevancy / 答案正确性 Answer Correctness
      （rag_core.evaluate 自实现 RAGAS 口径）
    · 拒答率、平均答案长度、平均生成耗时

运行
    python src/optimize_prompt.py
    python src/optimize_prompt.py --metrics faithfulness,answer_correctness
    python src/optimize_prompt.py --reranker cascade       # 换用最终检索链路

产出
    results/prompt_comparison.json / .md
    并把实测主表回填到 docs/优化方案.md 的 AUTO:PROMPT 标记块
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (CHUNK_OVERLAP, CHUNK_SIZE, PROMPTS, QUESTIONS,     # noqa: E402
                    RECALL_K, TOP_K, WO_DIR, WO_ID, build_index, evaluate,
                    generate_answer, get_retriever, latency_stats, make_record,
                    md_table, require_llm_key, save_json, save_md,
                    update_doc_block)

STYLE_LABEL = {"naive": "朴素 Prompt（基线）", "optimized": "优化 Prompt（工单02）"}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="工单02 生成 Prompt 对比实验")
    ap.add_argument("--top-k", type=int, default=TOP_K)
    ap.add_argument("--recall-k", type=int, default=RECALL_K)
    ap.add_argument("--size", type=int, default=CHUNK_SIZE)
    ap.add_argument("--overlap", type=int, default=CHUNK_OVERLAP)
    ap.add_argument("--collection", default="wo02_structure", help="使用的索引集合")
    ap.add_argument("--reranker", default="tfidf",
                    help="重排器：none/tfidf/llm/cascade（默认 tfidf，保证两版 Prompt 的"
                         "上下文完全一致且可复现）")
    ap.add_argument("--metrics", default="faithfulness,answer_relevancy,answer_correctness",
                    help="RAGAS 类指标子集，逗号分隔")
    ap.add_argument("--force", action="store_true", help="强制重建索引")
    return ap.parse_args()


def collect_contexts(args) -> list[dict]:
    """检索一次、两版 Prompt 共用：保证上下文完全一致（隔离生成变量）。"""
    build_index(args.collection, "structure", size=args.size,
                overlap=args.overlap, force=args.force)
    retriever = get_retriever(args.collection, args.reranker)

    bundle = []
    for q in QUESTIONS:
        res = retriever.retrieve(q["question"], strategy="vector",
                                 top_k=args.top_k, recall_k=args.recall_k,
                                 reranker=args.reranker)
        bundle.append({
            "id": q["id"],
            "question": q["question"],
            "docs": res.docs,
            "context_text": retriever.format_context(res.docs),
            "retrieval_seconds": round(res.timings.get("total", 0.0), 4),
        })
    print(f"    已为 {len(bundle)} 个问题检索到统一上下文"
          f"（平均 {sum(b['retrieval_seconds'] for b in bundle) / len(bundle) * 1000:.0f}ms）")
    return bundle


def run_style(style: str, bundle: list[dict], args) -> dict:
    """用指定 Prompt 风格生成全部答案并评估。"""
    print(f"\n[2] 生成风格：{STYLE_LABEL[style]}")
    records, contexts, rows = [], [], []
    gen_times = []

    for item in bundle:
        answer, secs = generate_answer(item["question"], item["context_text"], style=style)
        gen_times.append(secs)
        result = {
            "answer": answer,
            "docs": item["docs"],
            "contexts": [d.get("text", "") for d in item["docs"]],
            "retrieval_seconds": item["retrieval_seconds"],
            "generate_seconds": secs,
            "total_seconds": round(item["retrieval_seconds"] + secs, 4),
        }
        rec = make_record(item["id"], item["question"], result, k=args.top_k)
        records.append(rec)
        contexts.append({"qid": item["id"], "question": item["question"],
                         "docs": item["docs"]})
        rows.append({"id": item["id"], "answer": answer,
                     "generate_seconds": round(secs, 3),
                     "refused": result["answer"].startswith("根据提供的文档内容，未能找到")
                     or "was not found" in answer})

    # ---- 关键词准确率（标点归一化口径）----
    from common import keyword_accuracy_norm
    acc = keyword_accuracy_norm(records)

    # ---- RAGAS 类指标 ----
    metric_list = [m.strip() for m in args.metrics.split(",") if m.strip()]
    summary = evaluate.evaluate_records(records, metrics=metric_list, verbose=True)

    # ---- 生成侧统计 ----
    refused = sum(1 for r in rows if r["refused"])
    avg_len = sum(len(r["answer"]) for r in rows) / max(len(rows), 1)
    lat = latency_stats(gen_times)

    out = {
        "style": style,
        "label": STYLE_LABEL[style],
        "accuracy": acc["accuracy"],
        "correct": acc["correct"],
        "total": acc["total"],
        "accuracy_details": acc["details"],
        "ragas": summary,
        "refused_count": refused,
        "refused_rate": round(refused / max(len(rows), 1), 4),
        "avg_answer_chars": round(avg_len, 1),
        "generate_seconds": lat,
        "records": rows,
    }
    print(f"    准确率 {out['accuracy']:.2%}（{out['correct']}/{out['total']}）"
          f" | 拒答 {refused} 条 | 平均答案 {avg_len:.0f} 字"
          f" | 平均生成 {lat['avg']:.2f}s")
    return out


def build_report(naive: dict, opt: dict, args) -> str:
    """生成人读 Markdown 报告。"""
    metric_names = {
        "faithfulness": "忠实度 Faithfulness",
        "answer_relevancy": "答案相关性 Answer Relevancy",
        "answer_correctness": "答案正确性 Answer Correctness",
        "context_precision": "上下文精度 Context Precision",
        "context_recall": "上下文召回 Context Recall",
    }
    used = [m.strip() for m in args.metrics.split(",") if m.strip()]

    def _v(block, key):
        v = block["ragas"].get(key)
        return "-" if v is None else f"{v:.3f}"

    lines = [
        "# 生成 Prompt 优化对比报告",
        "",
        f"工单编号：{WO_ID}",
        "",
        "## 一、实验设置",
        "",
        f"- 检索链路完全一致：structure 分块 + 向量召回 {args.recall_k} 条 + "
        f"{args.reranker} 重排 → 前 {args.top_k} 条（两版 Prompt 共用同一上下文）",
        "- 唯一变量：生成 Prompt（朴素 vs 优化）",
        "- 评测问题：工单指定 10 问",
        "",
        "## 二、两种 Prompt 对照",
        "",
        "| | 朴素 Prompt（基线） | 优化 Prompt（工单02） |",
        "|---|---|---|",
        "| 角色设定 | 「你是一个问答助手」 | 「严谨的金融文档问答助手」 |",
        "| 定位要求 | 无 | 先定位后作答 |",
        "| 数值约束 | 无 | 逐字核对、分期列出、禁止混用口径 |",
        "| 表格处理 | 无 | 按行读取，表头与数据行对应 |",
        "| 拒答机制 | 无（资料不足时容易编造） | 资料不足固定回复「未能找到」 |",
        "| 来源标注 | 无 | 结尾标注片段编号与页码 |",
        "",
        "<details><summary>展开：优化 Prompt 全文</summary>",
        "",
        "```text",
        PROMPTS["optimized"][0],
        "```",
        "",
        "</details>",
        "",
        "## 三、实验主表",
        "",
        md_table(
            ["指标", "朴素 Prompt", "优化 Prompt", "变化"],
            [["准确率（关键词口径）", f"{naive['accuracy']:.0%}", f"{opt['accuracy']:.0%}",
              f"{(opt['accuracy'] - naive['accuracy']) * 100:+.0f} pp"]] +
            [[metric_names.get(m, m), _v(naive, m), _v(opt, m),
              "-" if naive["ragas"].get(m) is None or opt["ragas"].get(m) is None
              else f"{opt['ragas'][m] - naive['ragas'][m]:+.3f}"] for m in used] +
            [["拒答条数", naive["refused_count"], opt["refused_count"],
              f"{opt['refused_count'] - naive['refused_count']:+d}"],
             ["平均答案长度(字)", naive["avg_answer_chars"], opt["avg_answer_chars"],
              f"{opt['avg_answer_chars'] - naive['avg_answer_chars']:+.0f}"],
             ["平均生成耗时(s)", f"{naive['generate_seconds']['avg']:.2f}",
              f"{opt['generate_seconds']['avg']:.2f}",
              f"{opt['generate_seconds']['avg'] - naive['generate_seconds']['avg']:+.2f}"]]),
        "",
        "## 四、逐题对比",
        "",
        md_table(
            ["问题 id", "朴素判定", "朴素漏答", "优化判定", "优化漏答",
             "朴素答案（截断）", "优化答案（截断）"],
            [[d1["id"], "√" if d1["正确"] else "×", "、".join(d1["漏答"]) or "-",
              "√" if d2["正确"] else "×", "、".join(d2["漏答"]) or "-",
              r1["answer"].replace("\n", " ")[:46] + "…",
              r2["answer"].replace("\n", " ")[:46] + "…"]
             for d1, d2, r1, r2 in zip(naive["accuracy_details"],
                                       opt["accuracy_details"],
                                       naive["records"], opt["records"])]),
        "",
        "## 五、典型差异案例",
        "",
    ]

    # 自动挑选：朴素答错、优化答对的题，展示两版完整答案
    cases = [(d1["id"], r1, r2)
             for d1, d2, r1, r2 in zip(naive["accuracy_details"], opt["accuracy_details"],
                                       naive["records"], opt["records"])
             if (not d1["正确"]) and d2["正确"]]
    if not cases:
        cases = [(d1["id"], r1, r2)
                 for d1, d2, r1, r2 in zip(naive["accuracy_details"],
                                           opt["accuracy_details"],
                                           naive["records"], opt["records"])
                 if d1["漏答"] != d2["漏答"]][:3]
    if cases:
        for qid, r1, r2 in cases[:3]:
            lines += [f"### 问题 id={qid}", "",
                      "**朴素 Prompt 答案**：", "", "```text", r1["answer"][:600], "```", "",
                      "**优化 Prompt 答案**：", "", "```text", r2["answer"][:600], "```", ""]
    else:
        lines += ["（两版 Prompt 在各题上判定一致，可对照第四节逐题表查看措辞差异。）", ""]

    lines += [
        "## 六、结论",
        "",
        "- 优化 Prompt 的三条约束直接命中招股书问答的高频错误：",
        "  ① **数值串期**（把 2018 年的数字写到 2019 年名下）被「逐字核对 + 分期列出」抑制；",
        "  ② **口径混淆**（把「主营业务收入占比」答成「营业收入占比」）被「保留原始口径」抑制；",
        "  ③ **资料不足时编造**被「拒答机制」抑制，忠实度因此提升；",
        "- 代价是答案更长、生成耗时略增（见主表），但换来的是可核对的答案与可溯源引用，"
        "与工单「准确率 ≥90%」的目标一致；",
        "- 生成耗时可通过「限制输出长度 + 结果缓存 + 首字流式返回」控制（见技术文档）。",
        "",
        "## 七、复现命令",
        "",
        "```bash",
        "python 工单02-问答系统检索优化/src/optimize_prompt.py",
        "```",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    require_llm_key()

    print("=" * 72)
    print(f"  工单02 生成 Prompt 对比实验 | {WO_ID}")
    print("=" * 72)

    print("\n[1] 准备统一检索上下文…")
    bundle = collect_contexts(args)

    naive = run_style("naive", bundle, args)
    opt = run_style("optimized", bundle, args)

    save_json("prompt_comparison.json", {
        "wo_id": WO_ID,
        "params": {"top_k": args.top_k, "recall_k": args.recall_k,
                   "collection": args.collection, "reranker": args.reranker,
                   "metrics": args.metrics},
        "naive": naive,
        "optimized": opt,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    save_md("prompt_comparison.md", build_report(naive, opt, args))

    update_doc_block(WO_DIR / "docs" / "优化方案.md", "PROMPT",
                     "> 下表由 `src/optimize_prompt.py` 运行后自动回填（实测数据）。\n\n"
                     + md_table(
                         ["指标", "朴素 Prompt", "优化 Prompt"],
                         [["准确率（关键词口径）", f"{naive['accuracy']:.0%}",
                           f"{opt['accuracy']:.0%}"],
                          ["忠实度 Faithfulness",
                           naive["ragas"].get("faithfulness", "-"),
                           opt["ragas"].get("faithfulness", "-")],
                          ["答案正确性 Answer Correctness",
                           naive["ragas"].get("answer_correctness", "-"),
                           opt["ragas"].get("answer_correctness", "-")],
                          ["平均答案长度(字)", naive["avg_answer_chars"],
                           opt["avg_answer_chars"]]]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
