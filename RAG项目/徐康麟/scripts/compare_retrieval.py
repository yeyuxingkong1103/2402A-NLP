"""对比两份检索侧报告（``eval_retrieval.py --out`` 产物），逐题列出**翻转**。

为什么需要它：单看 Recall@5 的 ±0.03 分不清"真改善"还是"噪声"—— 必须看**哪几题变了、
方向如何、以及变化是不是选择器挑出来的**（本项目的规矩：一条杠杆要么给出机制解释，
要么如实记为噪声内无效）。同一批题、同一 k 才可比，所以这里只做同 k 报告对比。

用法::

    python scripts/compare_retrieval.py --base eval/results/selector-baseline.json \
        --compare eval/results/selector-on.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_report(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rank_move(before: int | None, after: int | None) -> str:
    """一题的排名变化方向：``better`` / ``worse`` / ``same``（None = 没命中）。"""
    if before == after:
        return "same"
    if after is not None and (before is None or after < before):
        return "better"
    return "worse"


def compare_rankers(base: dict, compare: dict) -> list[dict]:
    """逐个重排器比：返回 ``[{name, base, compare, deltas, flips}]``。"""
    out: list[dict] = []
    for name, after in (compare.get("rankers") or {}).items():
        before = (base.get("rankers") or {}).get(name)
        if before is None:
            continue
        base_ranks = before.get("ranks") or {}
        after_ranks = after.get("ranks") or {}
        detail = {row["id"]: row for row in (after.get("detail") or [])}
        flips: list[dict] = []
        for item_id in sorted(set(base_ranks) | set(after_ranks)):
            was, now = base_ranks.get(item_id), after_ranks.get(item_id)
            move = rank_move(was, now)
            if move == "same":
                continue
            row = detail.get(item_id, {})
            flips.append({"id": item_id, "before": was, "after": now, "move": move,
                          "picks": row.get("picks", 0),
                          "pick_expected": row.get("pick_expected")})
        out.append({
            "name": name,
            "base": {key: before.get(key) for key in ("recall_at_k", "mrr", "questions")},
            "compare": {key: after.get(key) for key in ("recall_at_k", "mrr", "questions",
                                                        "picked_any", "picked_correct",
                                                        "picked_rescued")},
            "delta_recall": round((after.get("recall_at_k", 0.0) or 0.0)
                                  - (before.get("recall_at_k", 0.0) or 0.0), 4),
            "delta_mrr": round((after.get("mrr", 0.0) or 0.0) - (before.get("mrr", 0.0) or 0.0), 4),
            "flips": flips,
        })
    return out


def render(comparisons: list[dict], *, k: int = 5) -> str:
    lines: list[str] = []
    for entry in comparisons:
        before, after = entry["base"], entry["compare"]
        lines.append(f"=== 重排器 {entry['name']}  (k={k}，题数 {before.get('questions')})")
        lines.append(f"  Recall@{k}: {before.get('recall_at_k')} -> {after.get('recall_at_k')}"
                     f"  Δ={entry['delta_recall']:+.4f}")
        lines.append(f"  MRR     : {before.get('mrr')} -> {after.get('mrr')}"
                     f"  Δ={entry['delta_mrr']:+.4f}")
        if after.get("picked_any") is not None:
            lines.append(f"  选择器  : 挑出 {after.get('picked_any')} 题 / 挑中正确条 "
                         f"{after.get('picked_correct')} 题 / 其中原本不在 top-{k} "
                         f"{after.get('picked_rescued')} 题")
        better = [f for f in entry["flips"] if f["move"] == "better"]
        worse = [f for f in entry["flips"] if f["move"] == "worse"]
        lines.append(f"  翻转    : 变好 {len(better)} 题，变差 {len(worse)} 题")
        for label, group in (("变好", better), ("变差", worse)):
            for flip in group:
                marker = "" if flip["pick_expected"] is None else f" 选择器挑中(原榜第 {flip['pick_expected']} 名)"
                lines.append(f"    [{label}] {flip['id']}: {flip['before']} -> {flip['after']}"
                             f"（挑条 {flip['picks']}）{marker}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="对比两份检索侧报告（逐题翻转）")
    parser.add_argument("--base", required=True)
    parser.add_argument("--compare", required=True)
    args = parser.parse_args()

    base, compare = load_report(args.base), load_report(args.compare)
    comparisons = compare_rankers(base, compare)
    if not comparisons:
        print("!! 两份报告没有共同的重排器名，无法对比（看 --out 的 rankers 键）", file=sys.stderr)
        return 2
    k = int(compare.get("k") or base.get("k") or 5)
    print(render(comparisons, k=k))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
