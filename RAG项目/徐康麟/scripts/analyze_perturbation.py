#!/usr/bin/env python3
"""扰动复测的判读：某条臂的分流"优势"在**改写措辞**之后还在不在？

为什么必须有这一步（2026-09-28 的教训）：六臂那轮 v5d 的分流子集 26/29（基座 24/29）
看着超标尺，但**原样重跑证明不了任何东西** —— 同条件两窗口的臂读数逐位相同
（解码确定性）。唯一的检验方式是**扰动同题**（`scripts/perturb_questions.py` 生成探针集）。

判读口径（**先定好，避免事后找说法**）：

* **主判据**：探针集里「A 臂独过 − B 臂独过」的**净翻转**。
  注意分母是探针条数（87 = 29 题 × 3 种扰动），原集那 29 题上的 +2 若为真，
  应当在这里表现为同方向、同量级的净翻转。
* **分扰动类型**：三种扰动的差值若方向一致，才谈得上"稳健"；互相打架就是措辞敏感。
* **单题稳健性**：原集里只被某条臂修好的题，其扰动形态是否**全部/多数**通过。

用法：
    python scripts/analyze_perturbation.py --arm-a qwen4b --arm-b v5d \
        --tag-prefix perturb --out eval/results/perturbation-verdict.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from legal_rag.logging_setup import console_safe  # noqa: E402


def load(results_dir: Path, prefix: str, arm: str) -> dict[str, dict]:
    path = results_dir / f"{prefix}-{arm}.jsonl"
    if not path.is_file():
        return {}
    handle = path.open(encoding="utf-8")
    with handle:
        return {str(json.loads(line).get("id")): json.loads(line)
                for line in handle if line.strip()}


def passed(record: dict | None) -> bool:
    return bool(((record or {}).get("score") or {}).get("ok"))


def probe_of(item_id: str) -> str:
    return item_id.split("~", 1)[1] if "~" in item_id else "original"


def source_of(item_id: str) -> str:
    return item_id.split("~", 1)[0]


def verdict(data_a: dict, data_b: dict, arm_a: str, arm_b: str) -> dict:
    ids = sorted(set(data_a) | set(data_b))
    green = [i for i in ids if passed(data_b.get(i)) and not passed(data_a.get(i))]
    red = [i for i in ids if passed(data_a.get(i)) and not passed(data_b.get(i))]
    by_probe: dict[str, dict] = {}
    for probe in sorted({probe_of(i) for i in ids}):
        subset = [i for i in ids if probe_of(i) == probe]
        by_probe[probe] = {
            "total": len(subset),
            arm_a: sum(1 for i in subset if passed(data_a.get(i))),
            arm_b: sum(1 for i in subset if passed(data_b.get(i))),
            "delta": (sum(1 for i in subset if passed(data_b.get(i)))
                      - sum(1 for i in subset if passed(data_a.get(i)))),
        }
    # 逐题：原题被 B 独好的那些，在扰动形态下如何
    focus: dict[str, dict] = {}
    for item in ids:
        source = source_of(item)
        if source in {source_of(i) for i in green}:
            focus.setdefault(source, {})[probe_of(item)] = {
                "arm_a": passed(data_a.get(item)), "arm_b": passed(data_b.get(item))}
    deltas = [entry["delta"] for probe, entry in by_probe.items() if probe != "original"]
    per_type_deltas = [entry["delta"] for probe, entry in by_probe.items()
                       if probe != "original"]
    consistent = bool(deltas) and (all(d > 0 for d in deltas) or all(d < 0 for d in deltas))
    return {
        "arm_a": arm_a, "arm_b": arm_b, "items": len(ids),
        "only_b": green, "only_a": red, "net_flip": len(green) - len(red),
        "by_probe": by_probe, "focus_items": focus,
        "per_type_deltas": per_type_deltas, "direction_consistent": consistent,
        "fails": {arm: dict(Counter(source_of(i) for i in ids if not passed(rows.get(i))))
                  for arm, rows in ((arm_a, data_a), (arm_b, data_b))},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="扰动复测判读")
    parser.add_argument("--results-dir", default=str(ROOT / "eval" / "results"))
    parser.add_argument("--tag-prefix", default="perturb")
    parser.add_argument("--arm-a", default="qwen4b", help="基线臂")
    parser.add_argument("--arm-b", default="v5d", help="对比臂")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    data_a = load(Path(args.results_dir), args.tag_prefix, args.arm_a)
    data_b = load(Path(args.results_dir), args.tag_prefix, args.arm_b)
    if not data_a or not data_b:
        print(f"缺少 {args.tag_prefix}-{{{args.arm_a},{args.arm_b}}}.jsonl（复测还没跑？）")
        return 1

    result = verdict(data_a, data_b, args.arm_a, args.arm_b)
    print(f"探针 {result['items']} 条；净翻转 = {result['net_flip']:+d} 条"
          f"（{args.arm_b} 独过 {len(result['only_b'])} / {args.arm_a} 独过 {len(result['only_a'])}）")
    for probe, entry in result["by_probe"].items():
        print(f"  {probe:14s} {args.arm_a} {entry[args.arm_a]:2d}/{entry['total']}  "
              f"{args.arm_b} {entry[args.arm_b]:2d}/{entry['total']}  差 {entry['delta']:+d}")
    print(f"  三种扰动方向一致？{result['direction_consistent']}  各类型差值 "
          f"{result['per_type_deltas']}")
    for item, forms in result["focus_items"].items():
        detail = "  ".join(f"{probe}=[{args.arm_a}:{'过' if forms[probe]['arm_a'] else '不过'}"
                           f" {args.arm_b}:{'过' if forms[probe]['arm_b'] else '不过'}]"
                           for probe in sorted(forms))
        print(f"  重点题 {item}: {console_safe(detail)}")
    if result["only_b"]:
        print(f"  {args.arm_b} 独过：" + console_safe(", ".join(result["only_b"])))
    if result["only_a"]:
        print(f"  {args.arm_a} 独过：" + console_safe(", ".join(result["only_a"])))

    if args.out:
        target = Path(args.out)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(result, ensure_ascii=False, indent=1) + "\n")
        print(f"已写：{target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
