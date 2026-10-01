# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
# 工单01 - 基于PDF文档的问答系统
"""
命令行评估：10 题 RAG vs 纯 LLM，三轨指标，输出 JSON + CSV。

  python scripts/eval.py                # 全量 10 题，含 LLM judge
  python scripts/eval.py --no-judge     # only 规则化命中（快，不调 judge）
  python scripts/eval.py --limit 3      # 先跑 3 题试水

【演示口径】规则化数值命中是最有说服力的一轨：
「RAG 命中 5/5 个数值题，纯 LLM 命中 0/5」是硬事实，
比「LLM 打了 0.87 分」更经得起追问。
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.evaluator import Evaluator  # noqa: E402


async def main() -> int:
    ap = argparse.ArgumentParser(description="工单01 · 10 题对比评估")
    ap.add_argument("--limit", type=int, default=None, help="只跑前 N 题")
    ap.add_argument("--no-judge", action="store_true", help="跳过 LLM judge")
    ap.add_argument("--no-norag", action="store_true", help="跳过纯 LLM 对照")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    ev = Evaluator()

    def progress(i: int, total: int, item) -> None:
        if args.quiet:
            return
        mark = "✅" if item.rule_hit else "❌"
        print(f"  [{i}/{total}] id={item.id} {mark} {item.rule_detail}"
              f"{('  错误：' + item.error) if item.error else ''}", flush=True)

    print("开始评估（RAG + 纯LLM + judge）…")
    report = await ev.evaluate_all(
        limit=args.limit,
        with_no_rag=not args.no_norag,
        with_judge=not args.no_judge,
        progress=progress,
    )

    s = report["summary"]
    print("\n" + "=" * 62)
    print("  评估结果")
    print("=" * 62)
    print(f"  题量                         {s['n']}")
    print(f"  规则化数值命中（RAG）        {s['rag_rule_hits']}/{s['n']}"
          f"  ← 最可信的一轨")
    print(f"  规则化数值命中（纯 LLM）     {s['no_rag_rule_hits']}/{s['n']}")
    print(f"  上下文相关性                 {s['avg_context_relevance']}")
    print(f"  忠实度（无幻觉）             {s['avg_faithfulness']}")
    print(f"  答案相关性                   {s['avg_answer_relevance']}")
    print(f"  上下文召回                   {s['avg_context_recall']}")
    print(f"  答案正确性                   {s['avg_answer_correctness']}")
    print(f"  TTFT   P50 / P95             {s['ttft_p50_ms']} / {s['ttft_p95_ms']} ms")
    print(f"  完整答案 P50 / P95           {s['total_p50_ms']} / {s['total_p95_ms']} ms")

    paths = Evaluator.save_report(report)
    print(f"\n  报告：{paths['json']}")
    print(f"        {paths['csv']}")

    print("\n  逐题（规则命中 / 纯LLM命中）：")
    for it in report["items"]:
        print(f"    id={it['id']:<4} {'✅' if it['rule_hit'] else '❌'}"
              f"  {'✅' if it.get('no_rag_rule_hit') else '❌'}"
              f"   {it['question'][:44]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
