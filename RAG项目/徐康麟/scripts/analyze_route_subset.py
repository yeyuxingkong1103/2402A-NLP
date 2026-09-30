#!/usr/bin/env python3
"""跨窗口看**分流子集**（must_refuse + must_route_general）的稳定性，并落成证据 JSON。

为什么要这个脚本：某一条臂在某一个窗口里"分流多过 2 题"，看着像超标尺的改善，
但**同一条臂换个窗口自己就会动** —— 不把这个抖动量出来，就会把噪声当结论
（本项目已经在召回 A/B 上栽过两次）。

它同时回答一个问题：**"原样重跑一次"能不能坐实？**
答：解码是确定性的（同条件两窗口的臂读数逐位相同）⇒ 原样重跑得到同样的答案，
不提供新信息。要真正复测必须**扰动**（同题改写/换会话/换窗口）或**扩大该维度样本**。

用法：
    python scripts/analyze_route_subset.py --out eval/results/route-subset-stability.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from legal_rag.logging_setup import console_safe  # noqa: E402

#: 窗口前缀 → 该窗口跑过的臂。加新窗口时在这里补一行即可。
WINDOWS: dict[str, tuple[str, ...]] = {
    "arm": ("qwen4b", "v1", "v5a", "v5b"),
    "arm2": ("qwen4b", "v1", "v5a", "v5b", "v5c"),
    "arm3": ("qwen4b", "v1", "v5a", "v5b", "v5c", "v5d"),
}
#: 分流子集里最难的几题（人工挑的，便于逐题看）
HARD_ITEMS = ("A04", "A11", "A12", "G07", "G10", "G16")
ROUTE_EXPECT = ("must_refuse", "must_route_general")


def load_window(results_dir: Path, window: str, arm: str) -> dict[str, dict] | None:
    path = results_dir / f"{window}-{arm}.jsonl"
    if not path.is_file():
        return None
    handle = path.open(encoding="utf-8")
    with handle:
        return {str(json.loads(line).get("id")): json.loads(line)
                for line in handle if line.strip()}


def passed(record: dict | None) -> bool:
    return bool(((record or {}).get("score") or {}).get("ok"))


def build(results_dir: Path) -> dict:
    report: dict = {"windows": {k: list(v) for k, v in WINDOWS.items()}, "route": {},
                    "hard_items": {}, "notes": []}
    route_ids: list[str] = []
    for window, arms in WINDOWS.items():
        for arm in arms:
            data = load_window(results_dir, window, arm)
            if data is None:
                continue
            if not route_ids:
                route_ids = [i for i, row in data.items()
                             if str(row.get("expect")) in ROUTE_EXPECT]
            key = f"{window}-{arm}"
            report["route"][key] = {
                "passed": sum(1 for i in route_ids if passed(data.get(i))),
                "total": len(route_ids),
            }
    report["route_total"] = len(route_ids)

    for item in HARD_ITEMS:
        report["hard_items"][item] = {}
        for window, arms in WINDOWS.items():
            for arm in arms:
                data = load_window(results_dir, window, arm)
                if data is not None and item in data:
                    report["hard_items"][item][f"{window}-{arm}"] = passed(data.get(item))

    # 同一条臂跨窗口的最大摆动 = 这个子集的"自抖动"，是判读 +2 题是否可信的标尺
    swing: dict[str, int] = {}
    for arm in ("qwen4b", "v1", "v5a", "v5b", "v5c", "v5d"):
        values = [entry["passed"] for key, entry in report["route"].items()
                  if key.endswith(f"-{arm}")]
        if len(values) >= 2:
            swing[arm] = max(values) - min(values)
    report["same_arm_swing"] = swing
    if swing:
        report["notes"].append(
            f"同一条臂跨窗口在本子集上的最大摆动 = {max(swing.values())} 题"
            f"（{swing}）⇒ 小于等于这个数的臂间差不可辨。")
    report["notes"].append(
        "解码是确定性的（同条件两窗口读数逐位相同）⇒ **原样重跑不提供新信息**；"
        "要复测必须扰动（同题改写/换会话）或扩大该维度样本量。")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="分流子集跨窗口稳定性")
    parser.add_argument("--results-dir", default=str(ROOT / "eval" / "results"))
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    report = build(Path(args.results_dir))
    text = json.dumps(report, ensure_ascii=False, indent=1)

    print(f"分流子集共 {report['route_total']} 题；逐（窗口,臂）通过数：")
    for key, entry in report["route"].items():
        print(f"  {key:14s} {entry['passed']}/{entry['total']}")
    print(f"同一条臂跨窗口摆动：{report['same_arm_swing']}")
    for note in report["notes"]:
        # 打印走 console_safe：GBK 控制台打不出 ⇒ 这类符号会 UnicodeEncodeError
        # （静态闸门只管网字面量，管不到这里），JSON 里保留原样。
        print(f"  [注意] {console_safe(note)}")

    if args.out:
        target = Path(args.out)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(text + "\n")
        print(f"已写：{target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
