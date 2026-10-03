# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化
# 工单01 - 基于PDF文档的问答系统
# 工单02 - 基于PDF文档的问答系统的优化
"""
命令行评估：10 题 RAG vs 纯 LLM，三轨指标，输出 JSON + CSV。

  # —— 工单01 的用法（仍在用，行为不变）——
  python scripts/eval.py                # 全量 10 题，含 LLM judge
  python scripts/eval.py --no-judge     # only 规则化命中（快，不调 judge）
  python scripts/eval.py --limit 3      # 先跑 3 题试水

  # —— 工单02 新增：按检索剖面跑 / 三剖面一次性对比 ——
  python scripts/eval.py --profile optimized          # 单剖面
  python scripts/eval.py --compare baseline,delivered,optimized
  python scripts/eval.py --compare all                # 等价于上面三个

【演示口径】规则化数值命中是最有说服力的一轨：
「RAG 命中 5/5 个数值题，纯 LLM 命中 0/5」是硬事实，
比「LLM 打了 0.87 分」更经得起追问。

【工单02 追加的口径】对比表里还有一列 `CKC`（上下文关键词覆盖）——
它是**纯检索侧**指标：答案关键词在不在召回的上下文里。
rule_hit 是端到端的（掺了生成的成分），CKC 才是"检索精确度"的干净度量。
两者一起看，才能分清"检索没召回"和"召回了但模型没答对"。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings                                          # noqa: E402
from app.core.evaluator import Evaluator                                 # noqa: E402
from app.core.profiles import PROFILES, get_profile                      # noqa: E402


def _fmt_pct(v) -> str:
    return "  -  " if v is None else f"{v * 100:5.1f}%"


def print_table(rows: list[dict]) -> None:
    """打印剖面对比表 —— 这张表就是工单02「优化前后检索精确度变化」的产物。"""
    head = (f"{'剖面':<12}{'规则命中':>10}{'CKC':>8}{'页精确':>8}{'页召回':>8}"
            f"{'上下文':>8}{'TTFT-P50':>10}{'TTFT-P95':>10}{'≤3s':>7}"
            f"{'合并':>6}{'邻块':>6}")
    print("\n" + "=" * (len(head) + 6))
    print("  工单02 · 检索剖面横向对比（10 题）")
    print("=" * (len(head) + 6))
    print(head)
    print("-" * (len(head) + 6))
    for r in rows:
        s = r["summary"]
        under = s.get("ttft_under_3s", 0)
        print(f"{s['profile']:<12}"
              f"{str(s['rag_rule_hits']) + '/' + str(s['n']):>10}"
              f"{_fmt_pct(s.get('avg_context_keyword_coverage')):>8}"
              f"{_fmt_pct(s.get('avg_page_precision')):>8}"
              f"{_fmt_pct(s.get('avg_page_recall')):>8}"
              f"{s.get('avg_context_chars', 0):>8}"
              f"{str(s['ttft_p50_ms']) + 'ms':>10}"
              f"{str(s['ttft_p95_ms']) + 'ms':>10}"
              f"{str(under) + '/' + str(s['n']):>7}"
              f"{s.get('n_merged_total', 0):>6}"
              f"{s.get('n_added_neighbors_total', 0):>6}")
    print("-" * (len(head) + 6))
    print("说明：CKC / 页精确 / 页召回 是纯检索侧指标（邻块不计入分母）；")
    print("      规则命中是端到端指标。两者一起看，才能区分「检索漏了」与「生成漏了」。")
    print("      ⚠️ 时间口径：n=10 时 **P95 就等于 10 个样本里的最大值**，单个离群点")
    print("         就能把它拉爆（实测同配置重跑会差 1 秒以上）。所以「≤3 秒」看")
    print("         **P50 与 ≤3s 比例**，P95 作为上界一并披露，不作为唯一判据。")


async def run_one(profile_name: str, args, quiet: bool) -> dict:
    ev = Evaluator(profile=get_profile(profile_name))

    def progress(i: int, total: int, item) -> None:
        if quiet:
            return
        mark = "✅" if item.rule_hit else "❌"
        print(f"  [{i}/{total}] {profile_name} id={item.id} {mark}"
              f" CKC={_fmt_pct(item.context_keyword_coverage)}"
              f" {item.rule_detail[:40]}"
              f"{('  错误：' + item.error) if item.error else ''}", flush=True)

    return await ev.evaluate_all(
        limit=args.limit,
        with_no_rag=not args.no_norag,
        with_judge=not args.no_judge,
        progress=progress,
    )


async def main() -> int:
    ap = argparse.ArgumentParser(
        description="工单02 · 10 题对比评估（支持检索剖面）")
    ap.add_argument("--limit", type=int, default=None, help="只跑前 N 题")
    ap.add_argument("--no-judge", action="store_true", help="跳过 LLM judge")
    ap.add_argument("--no-norag", action="store_true", help="跳过纯 LLM 对照")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--profile", default=None,
                    help=f"用哪个检索剖面跑：{'/'.join(PROFILES)}")
    ap.add_argument("--compare", default=None,
                    help="多剖面对比，逗号分隔；`all` = 全部剖面。"
                         "对比模式默认跳过纯 LLM 对照（它与剖面无关，重复跑纯属浪费）")
    args = ap.parse_args()

    # ---------- 对比模式 ----------
    if args.compare:
        names = (list(PROFILES) if args.compare.strip().lower() == "all"
                 else [n.strip() for n in args.compare.split(",") if n.strip()])
        bad = [n for n in names if n not in PROFILES]
        if bad:
            print(f"未知剖面 {bad}，可选：{list(PROFILES)}", file=sys.stderr)
            return 2
        # 纯 LLM 基线与检索剖面无关，对比时默认跳过（否则每个剖面白跑 10 次 LLM）
        if len(names) > 1:
            args.no_norag = True

        reports = []
        for n in names:
            print(f"\n>>> 剖面 {n} …", flush=True)
            reports.append(await run_one(n, args, args.quiet))

        # 每个剖面单独存档、互不覆盖（带 tag 时 save_report 不动 eval-latest.json）
        paths = [Evaluator.save_report(r, tag=r["summary"]["profile"])
                 for r in reports]

        print_table(reports)

        out = settings.data_path / "eval"
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        combined = {
            "generated_at": stamp,
            "profiles": names,
            "summaries": {r["summary"]["profile"]: r["summary"] for r in reports},
            "items": {r["summary"]["profile"]: r["items"] for r in reports},
        }
        cp = out / f"compare-{stamp}.json"
        cp.write_text(json.dumps(combined, ensure_ascii=False, indent=2),
                      encoding="utf-8")
        (out / "compare-latest.json").write_text(
            json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n  各剖面报告：{paths[0]['json']} 等 {len(paths)} 份")
        print(f"  对比报告：  {cp}")
        return 0

    # ---------- 单剖面模式（工单01 的行为保持不变）----------
    name = args.profile or settings.retrieval_profile
    print(f"开始评估（剖面 {name}｜RAG + 纯LLM + judge）…")
    report = await run_one(name, args, args.quiet)

    s = report["summary"]
    print("\n" + "=" * 62)
    print(f"  评估结果（剖面：{s['profile']}）")
    print("=" * 62)
    print(f"  题量                         {s['n']}")
    print(f"  规则化数值命中（RAG）        {s['rag_rule_hits']}/{s['n']}"
          f"  ← 最可信的一轨")
    print(f"  规则化数值命中（纯 LLM）     {s['no_rag_rule_hits']}/{s['n']}")
    print("  ── 检索侧（工单02）──")
    print(f"  上下文关键词覆盖 CKC         {s.get('avg_context_keyword_coverage')}")
    print(f"  检索页精确率                 {s.get('avg_page_precision')}")
    print(f"  检索页召回率                 {s.get('avg_page_recall')}")
    print(f"  平均上下文字数               {s.get('avg_context_chars')}")
    print(f"  同事实合并 / 邻块补充        {s.get('n_merged_total')} / "
          f"{s.get('n_added_neighbors_total')}")
    print("  ── 端到端 ──")
    print(f"  上下文相关性                 {s['avg_context_relevance']}")
    print(f"  忠实度（无幻觉）             {s['avg_faithfulness']}")
    print(f"  答案相关性                   {s['avg_answer_relevance']}")
    print(f"  上下文召回                   {s['avg_context_recall']}")
    print(f"  答案正确性                   {s['avg_answer_correctness']}")
    print(f"  TTFT   P50 / P95 / 最大      {s['ttft_p50_ms']} / {s['ttft_p95_ms']}"
          f" / {s.get('ttft_max_ms', 0)} ms")
    print(f"  TTFT   ≤3s 的题数            {s.get('ttft_under_3s', 0)}/{s['n']}"
          f"   ← 比 P95 更稳的达标判据（n=10 时 P95=最大值）")
    print(f"  完整答案 P50 / P95           {s['total_p50_ms']} / {s['total_p95_ms']} ms")

    paths = Evaluator.save_report(report)
    print(f"\n  报告：{paths['json']}")
    print(f"        {paths['csv']}")

    print("\n  逐题（规则命中 / 纯LLM命中 / CKC）：")
    for it in report["items"]:
        print(f"    id={it['id']:<4} {'✅' if it['rule_hit'] else '❌'}"
              f"  {'✅' if it.get('no_rag_rule_hit') else '❌'}"
              f"  {_fmt_pct(it.get('context_keyword_coverage'))}"
              f"   {it['question'][:40]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
