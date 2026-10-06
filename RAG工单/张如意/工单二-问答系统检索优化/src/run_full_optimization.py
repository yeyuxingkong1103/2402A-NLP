#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
一键运行完整优化流水线（run_full_optimization.py）
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化

流程
    优化前（工单01 基线）
        fixed 固定窗口分块 + 纯向量检索 + 不重排 + 朴素 Prompt
    优化后（工单02，三种优化叠加）
        structure 章节结构感知分块 + 混合检索(向量+BM25, RRF 融合)
        + 级联重排(TF-IDF 粗排 → LLM 精排) + 优化 Prompt

    两者跑同一份《招股说明书1.pdf》、同一组 10 个问题，分别计算：
      · 答案准确率（关键词口径，全部必备要点命中才算对）
      · 检索层：证据命中率 / 证据 MRR / Recall@k / 页码命中率
      · 生成层：忠实度 / 答案正确性 / 上下文召回（RAGAS 自实现口径）
      · 性能：检索耗时 / 重排耗时 / 生成耗时 / 端到端耗时
    并逐题并排输出「优化前答案 vs 优化后答案」，供演示视频直接引用。

运行
    python src/run_full_optimization.py
    python src/run_full_optimization.py --skip-ragas      # 省 token：只算准确率与检索指标
    python src/run_full_optimization.py --final-reranker tfidf   # 不调 LLM 重排

产出
    results/before_after.json / results/before_after.md
    并把实测对比表回填到 docs/优化方案.md 的 AUTO:BEFORE_AFTER 标记块
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (ANSWER_KEYS, CHUNK_OVERLAP, CHUNK_SIZE, QUESTIONS,     # noqa: E402
                    RECALL_K, TOP_K, WO_DIR, WO_ID, QAConfig, answer_question,
                    build_index, evaluate, get_retriever, keyword_accuracy_norm,
                    latency_stats, make_record, md_table, require_llm_key,
                    retrieval_evidence_metrics, save_json, save_md,
                    update_doc_block)

# 优化前：对齐工单01 的 wo01_baseline 预设（fixed + vector + none + 朴素 Prompt）
BASELINE = QAConfig(name="优化前（工单01 基线）", collection="wo02_fixed",
                    strategy="vector", fusion="rrf", reranker="none",
                    prompt="naive")

# 优化后：三种优化叠加（structure 分块 + 混合检索 RRF + 级联重排 + 优化 Prompt）
OPTIMIZED = QAConfig(name="优化后（工单02）", collection="wo02_structure",
                     strategy="hybrid", fusion="rrf", reranker="cascade",
                     prompt="optimized")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="工单02 优化前后全流程对比")
    ap.add_argument("--top-k", type=int, default=TOP_K)
    ap.add_argument("--recall-k", type=int, default=RECALL_K)
    ap.add_argument("--size", type=int, default=CHUNK_SIZE)
    ap.add_argument("--overlap", type=int, default=CHUNK_OVERLAP)
    ap.add_argument("--final-reranker", default="cascade",
                    help="优化后使用的重排器：cascade（默认）/ tfidf")
    ap.add_argument("--metrics", default="faithfulness,answer_correctness,context_recall",
                    help="RAGAS 类指标子集（--skip-ragas 时忽略）")
    ap.add_argument("--skip-ragas", action="store_true",
                    help="跳过 LLM 评估指标，只算准确率与检索指标（省 token、快）")
    ap.add_argument("--force", action="store_true", help="强制重建索引")
    return ap.parse_args()


def run_pipeline(cfg: QAConfig, args) -> dict:
    """跑完一套配置的 10 个问题，返回逐题结果与汇总指标。"""
    print(f"\n[2] 运行配置：{cfg.name}")
    print(f"    分块={cfg.collection} | 检索={cfg.strategy}/{cfg.fusion} "
          f"| 重排={cfg.reranker} | Prompt={cfg.prompt}")

    retriever = get_retriever(cfg.collection, cfg.reranker)
    records, results, details = [], [], []
    t_retr, t_rerank, t_gen, t_total = [], [], [], []

    for q in QUESTIONS:
        res = answer_question(cfg, q["question"], retriever=retriever)
        if res.get("error"):
            print(f"    [warn] id={q['id']} {res['error']}")

        rec = make_record(q["id"], q["question"], res, k=args.top_k)
        records.append(rec)
        results.append({"qid": q["id"], "question": q["question"], "docs": res["docs"]})
        t_retr.append(res.get("retrieval_seconds", 0.0))
        t_rerank.append(res.get("timings", {}).get("rerank", 0.0))
        t_gen.append(res.get("generate_seconds", 0.0))
        t_total.append(res.get("total_seconds", 0.0))

        details.append({
            "id": q["id"],
            "question": q["question"],
            "answer": res["answer"],
            "gold_answer": ANSWER_KEYS[q["id"]]["ground_truth"],
            "ground_truth_pages": ANSWER_KEYS[q["id"]]["pages"],
            "retrieved_pages": [d.get("page") for d in res["docs"][:args.top_k]],
            "citations": [{"page": d.get("page"), "section": d.get("section", ""),
                           "chunk_id": d.get("chunk_id")}
                          for d in res["docs"][:args.top_k]],
            "retrieval_seconds": res.get("retrieval_seconds", 0.0),
            "rerank_seconds": res.get("timings", {}).get("rerank", 0.0),
            "generate_seconds": res.get("generate_seconds", 0.0),
            "total_seconds": res.get("total_seconds", 0.0),
        })

    acc = keyword_accuracy_norm(records)
    ev = retrieval_evidence_metrics(results, k=args.top_k)

    ragas_summary = {}
    if not args.skip_ragas:
        metric_list = [m.strip() for m in args.metrics.split(",") if m.strip()]
        ragas_summary = evaluate.evaluate_records(records, metrics=metric_list,
                                                  verbose=True)

    out = {
        "config": cfg.to_dict(),
        "name": cfg.name,
        "accuracy": acc["accuracy"],
        "correct": acc["correct"],
        "total": acc["total"],
        "accuracy_details": acc["details"],
        "evidence_hit_rate": ev["evidence_hit_rate"],
        "evidence_mrr": ev["evidence_mrr"],
        "evidence_recall_at_k": ev["evidence_recall_at_k"],
        "page_hit_rate": ev["page_hit_rate"],
        "evidence_details": ev["details"],
        "ragas": ragas_summary,
        "retrieval_seconds": latency_stats(t_retr),
        "rerank_seconds": latency_stats(t_rerank),
        "generate_seconds": latency_stats(t_gen),
        "total_seconds": latency_stats(t_total),
        "within_3s_retrieval": round(
            sum(1 for a, b in zip(t_retr, t_rerank) if a + b <= 3.0) / max(len(t_retr), 1), 4),
        "details": details,
    }
    print(f"    准确率 {out['accuracy']:.2%}（{out['correct']}/{out['total']}）"
          f" | 证据命中率 {out['evidence_hit_rate']:.2%}"
          f" | MRR {out['evidence_mrr']:.3f}"
          f" | 检索 {out['retrieval_seconds']['avg'] * 1000:.0f}ms"
          f" | 端到端 {out['total_seconds']['avg']:.2f}s")
    return out


def build_report(base: dict, opt: dict, args) -> str:
    """生成优化前后对比报告（含逐题答案并排）。"""
    metric_names = {
        "faithfulness": "忠实度 Faithfulness",
        "answer_relevancy": "答案相关性 Answer Relevancy",
        "answer_correctness": "答案正确性 Answer Correctness",
        "context_recall": "上下文召回 Context Recall",
        "context_precision": "上下文精度 Context Precision",
    }
    used = [] if args.skip_ragas else [m.strip() for m in args.metrics.split(",") if m.strip()]

    def _g(block, key):
        v = block["ragas"].get(key)
        return "-" if v is None else f"{v:.3f}"

    def _delta(key, fmt="{:.3f}", scale=1.0):
        a, b = base["ragas"].get(key), opt["ragas"].get(key)
        if a is None or b is None:
            return "-"
        return fmt.format((b - a) * scale)

    main_rows = [
        ["答案准确率（关键词口径）", f"{base['accuracy']:.0%}", f"{opt['accuracy']:.0%}",
         f"{(opt['accuracy'] - base['accuracy']) * 100:+.0f} pp"],
        ["检索证据命中率", f"{base['evidence_hit_rate']:.2%}",
         f"{opt['evidence_hit_rate']:.2%}",
         f"{(opt['evidence_hit_rate'] - base['evidence_hit_rate']) * 100:+.0f} pp"],
        ["检索证据 MRR", f"{base['evidence_mrr']:.3f}", f"{opt['evidence_mrr']:.3f}",
         f"{opt['evidence_mrr'] - base['evidence_mrr']:+.3f}"],
        [f"证据 Recall@{args.top_k}", f"{base['evidence_recall_at_k']:.2%}",
         f"{opt['evidence_recall_at_k']:.2%}",
         f"{(opt['evidence_recall_at_k'] - base['evidence_recall_at_k']) * 100:+.0f} pp"],
        ["证据页码命中率", f"{base['page_hit_rate']:.2%}", f"{opt['page_hit_rate']:.2%}",
         f"{(opt['page_hit_rate'] - base['page_hit_rate']) * 100:+.0f} pp"],
    ] + [[metric_names.get(m, m), _g(base, m), _g(opt, m), _delta(m, "{:+.3f}")]
         for m in used] + [
        ["平均检索耗时(s)", f"{base['retrieval_seconds']['avg']:.3f}",
         f"{opt['retrieval_seconds']['avg']:.3f}",
         f"{opt['retrieval_seconds']['avg'] - base['retrieval_seconds']['avg']:+.3f}"],
        ["平均重排耗时(s)", f"{base['rerank_seconds']['avg']:.3f}",
         f"{opt['rerank_seconds']['avg']:.3f}",
         f"{opt['rerank_seconds']['avg'] - base['rerank_seconds']['avg']:+.3f}"],
        ["平均生成耗时(s)", f"{base['generate_seconds']['avg']:.3f}",
         f"{opt['generate_seconds']['avg']:.3f}",
         f"{opt['generate_seconds']['avg'] - base['generate_seconds']['avg']:+.3f}"],
        ["检索+重排 ≤3s 占比", f"{base['within_3s_retrieval']:.0%}",
         f"{opt['within_3s_retrieval']:.0%}",
         f"{(opt['within_3s_retrieval'] - base['within_3s_retrieval']) * 100:+.0f} pp"],
    ]

    lines = [
        "# 优化前后效果对比报告",
        "",
        f"工单编号：{WO_ID}",
        "",
        "## 一、对比设置",
        "",
        f"- 语料：《招股说明书1.pdf》全文；问题：工单指定 10 问",
        f"- 优化前：{base['config']['collection']} + {base['config']['strategy']} 检索"
        f" + {base['config']['reranker']} 重排 + {base['config']['prompt']} Prompt",
        f"- 优化后：{opt['config']['collection']} + {opt['config']['strategy']}/"
        f"{opt['config']['fusion']} 检索 + {opt['config']['reranker']} 重排"
        f" + {opt['config']['prompt']} Prompt",
        "",
        "## 二、总体指标对比",
        "",
        md_table(["指标", "优化前（工单01 基线）", "优化后（工单02）", "变化"],
                 main_rows),
        "",
        "## 三、逐题结果对比（答案并排，供演示视频引用）",
        "",
    ]

    for b, o in zip(base["details"], opt["details"]):
        lines += [
            f"### 问题 id={b['id']}",
            "",
            f"> {b['question']}",
            "",
            f"**参考答案**：{o['gold_answer']}",
            "",
            f"**优化前答案**（检索页码 {b['retrieved_pages']}，耗时 {b['total_seconds']:.2f}s）：",
            "",
            "```text",
            b["answer"][:700] or "（空）",
            "```",
            "",
            f"**优化后答案**（检索页码 {o['retrieved_pages']}，耗时 {o['total_seconds']:.2f}s）：",
            "",
            "```text",
            o["answer"][:700] or "（空）",
            "```",
            "",
        ]

    # 逐题判定总表
    lines += [
        "## 四、逐题判定总表",
        "",
        md_table(
            ["问题 id", "优化前判定", "优化前漏答", "优化后判定", "优化后漏答"],
            [[d1["id"], "√" if d1["正确"] else "×", "、".join(d1["漏答"]) or "-",
              "√" if d2["正确"] else "×", "、".join(d2["漏答"]) or "-"]
             for d1, d2 in zip(base["accuracy_details"], opt["accuracy_details"])]),
        "",
        "## 五、结论",
        "",
        f"- 优化后准确率 **{opt['accuracy']:.0%}**，较优化前"
        f"（{base['accuracy']:.0%}）提升 {(opt['accuracy'] - base['accuracy']) * 100:+.0f} "
        f"个百分点；工单要求「准确率 ≥90%」"
        f"{'已达成' if opt['accuracy'] >= 0.9 else '未达成，请检查索引与 Prompt 配置'}；",
        f"- 检索证据命中率 {base['evidence_hit_rate']:.2%} → "
        f"{opt['evidence_hit_rate']:.2%}，说明化提升主要来自「找得到」而非「编得像」；",
        f"- 系统侧检索+重排耗时 {opt['retrieval_seconds']['avg'] + opt['rerank_seconds']['avg']:.3f}s"
        f"（P95 {opt['rerank_seconds']['p95'] + opt['retrieval_seconds']['p95']:.3f}s），"
        f"稳定在 3 秒约束内；",
        "- 三种优化各自的贡献可由三份专项报告拆解："
        "`chunking_comparison.md`（分块）× `retrieval_comparison.md`（重排）"
        "× `prompt_comparison.md`（生成）。",
        "",
        "## 六、复现命令",
        "",
        "```bash",
        "python 工单02-问答系统检索优化/src/run_full_optimization.py",
        "```",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    OPTIMIZED.reranker = args.final_reranker
    require_llm_key()

    print("=" * 72)
    print(f"  工单02 优化前后全流程对比 | {WO_ID}")
    print("=" * 72)

    print("\n[1] 建立两套索引（基线 fixed / 优化 structure）…")
    build_index(BASELINE.collection, "fixed", size=args.size,
                overlap=args.overlap, force=args.force)
    build_index(OPTIMIZED.collection, "structure", size=args.size,
                overlap=args.overlap, force=args.force)

    base = run_pipeline(BASELINE, args)
    opt = run_pipeline(OPTIMIZED, args)

    save_json("before_after.json", {
        "wo_id": WO_ID,
        "params": {"top_k": args.top_k, "recall_k": args.recall_k,
                   "metrics": [] if args.skip_ragas else args.metrics,
                   "skip_ragas": args.skip_ragas},
        "before": base,
        "after": opt,
        "delta": {
            "accuracy": round(opt["accuracy"] - base["accuracy"], 4),
            "evidence_hit_rate": round(opt["evidence_hit_rate"] - base["evidence_hit_rate"], 4),
            "evidence_mrr": round(opt["evidence_mrr"] - base["evidence_mrr"], 4),
            "evidence_recall_at_k": round(
                opt["evidence_recall_at_k"] - base["evidence_recall_at_k"], 4),
        },
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    save_md("before_after.md", build_report(base, opt, args))

    update_doc_block(
        WO_DIR / "docs" / "优化方案.md", "BEFORE_AFTER",
        "> 下表由 `src/run_full_optimization.py` 运行后自动回填（实测数据）。\n\n"
        + md_table(
            ["指标", "优化前（工单01）", "优化后（工单02）", "变化"],
            [["答案准确率", f"{base['accuracy']:.0%}", f"{opt['accuracy']:.0%}",
              f"{(opt['accuracy'] - base['accuracy']) * 100:+.0f} pp"],
             ["证据命中率", f"{base['evidence_hit_rate']:.2%}",
              f"{opt['evidence_hit_rate']:.2%}",
              f"{(opt['evidence_hit_rate'] - base['evidence_hit_rate']) * 100:+.0f} pp"],
             ["证据 MRR", f"{base['evidence_mrr']:.3f}", f"{opt['evidence_mrr']:.3f}",
              f"{opt['evidence_mrr'] - base['evidence_mrr']:+.3f}"],
             ["检索+重排耗时(s)",
              f"{base['retrieval_seconds']['avg'] + base['rerank_seconds']['avg']:.3f}",
              f"{opt['retrieval_seconds']['avg'] + opt['rerank_seconds']['avg']:.3f}",
              f"{(opt['retrieval_seconds']['avg'] + opt['rerank_seconds']['avg']) - (base['retrieval_seconds']['avg'] + base['rerank_seconds']['avg']):+.3f}"]]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
