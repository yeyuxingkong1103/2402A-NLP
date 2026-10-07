"""
评估脚本
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
工单编号：人工智能NLP-RAG-图像内容解析及检索优化

用法：
    python scripts/run_eval.py                        # 默认跑 data/eval/ticket_questions.json
    python scripts/run_eval.py --no-metrics           # 只出 RAG vs 纯LLM 的答案对比，不调评审模型
    python scripts/run_eval.py --dataset xxx.json     # 换评测集
    python scripts/run_eval.py --limit 3              # 先跑前 3 题试水

产出：
    data/eval/report_<时间戳>.json   机器可读
    data/eval/report_<时间戳>.md     人可读（含答案对比表）
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.evaluator import evaluate, load_questions  # noqa: E402


def to_markdown(report: dict) -> str:
    s = report["summary"]
    lines = [
        f"# RAG 评估报告",
        "",
        f"- 工单编号：{report['work_order_no']}",
        f"- 生成时间：{report['generated_at']}",
        f"- 用例数：{report['num_cases']}，总耗时：{report['elapsed_seconds']}s",
        "",
        "## 一、汇总指标",
        "",
        "| 指标 | 说明 | RAG | 纯 LLM |",
        "|---|---|---|---|",
        f"| 忠实度 faithfulness | 答案是否由检索上下文支撑 | {s.get('faithfulness')} | {s.get('llm_only_faithfulness')} |",
        f"| 答案相关性 answer_relevancy | 答案是否切题 | {s.get('answer_relevancy')} | {s.get('llm_only_relevancy')} |",
        f"| 上下文精确率 context_precision | 召回片段的命中比例 | {s.get('context_precision')} | — |",
        f"| 上下文召回率 context_recall | 参考答案要点被覆盖比例 | {s.get('context_recall')} | — |",
        f"| 答案正确性 answer_correctness | 与参考答案一致程度 | {s.get('answer_correctness')} | {s.get('llm_only_correctness')} |",
        "",
        "## 二、性能",
        "",
        f"- 平均响应：**{s.get('avg_total_ms')} ms**，最大：{s.get('max_total_ms')} ms",
        f"- ≤3 秒达标率：**{s.get('within_3s_ratio')}**",
        f"- 被阈值拦空（无依据）用例数：{s.get('gated_cases')}",
        "",
        "## 三、逐题结果",
        "",
    ]
    for c in report["cases"]:
        m = c.get("metrics") or {}
        lines.append(f"### id={c['id']} {c['question']}")
        lines.append("")
        if c.get("reference"):
            lines.append(f"**参考答案**：{c['reference']}")
            lines.append("")
        lines.append(f"**RAG 回答**（{c['timing'].get('total_ms')} ms，引用 {len(c['citations'])} 条）：")
        lines.append("")
        lines.append("> " + (c["rag_answer"] or "").replace("\n", "\n> "))
        lines.append("")
        lines.append("**纯 LLM 回答**：")
        lines.append("")
        lines.append("> " + (c["llm_only_answer"] or "").replace("\n", "\n> "))
        lines.append("")
        if m:
            parts = []
            for k, v in m.items():
                if isinstance(v, dict) and v.get("score") is not None:
                    parts.append(f"{k}={v['score']}")
            if parts:
                lines.append("**指标**：" + "；".join(parts))
                lines.append("")
        lines.append("---")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=config.EVAL_DATASET)
    ap.add_argument("--no-metrics", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--top-k", type=int, default=0)
    args = ap.parse_args()

    config.ensure_dirs()
    questions = load_questions(args.dataset)
    if args.limit:
        questions = questions[: args.limit]

    print(f"[工单] {config.WORK_ORDER_NO}")
    print(f"[评测] {args.dataset}，{len(questions)} 题，指标：{'关' if args.no_metrics else '开'}"
          f"，串行执行（避开账号级限流）\n")

    # 评估脚本必须先预热：否则第一题的 retrieve 要额外承担「索引加载 + jieba 建
    # BM25 表 + 向量模型加载」，实测会把这一题的耗时从 20ms 拉到 14 秒，
    # 污染 avg/max 口径（这不是检索慢，是冷启动）。
    t_warm = time.perf_counter()
    from src.embedder import embed_query
    from src.index_store import KnowledgeBase

    _kb = KnowledgeBase.get()
    _ = _kb.bm25
    embed_query("预热")
    print(f"[预热] 索引 {len(_kb.chunks)} 块，耗时 {time.perf_counter()-t_warm:.1f}s\n")

    report = evaluate(
        questions,
        with_metrics=not args.no_metrics,
        top_k=args.top_k or None,
    )

    stamp = time.strftime("%Y%m%d_%H%M%S")
    json_path = config.EVAL_DIR / f"report_{stamp}.json"
    md_path = config.EVAL_DIR / f"report_{stamp}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(to_markdown(report), encoding="utf-8")

    print("\n=== 汇总 ===")
    for k, v in report["summary"].items():
        print(f"  {k}: {v}")
    print(f"\n[OK] {json_path}")
    print(f"[OK] {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
