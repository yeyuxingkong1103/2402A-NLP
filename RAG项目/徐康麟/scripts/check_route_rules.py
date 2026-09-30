#!/usr/bin/env python3
"""在**冻结的 107 题评测集**上实测预路由规则的精度与召回（不需要 GPU）。

为什么必须先量再接线：预路由一旦误伤，法律题就会**不检索资料、引用为空**——
代价远大于漏掉一条闲聊题。所以判据先定好：

* **精度（硬门槛）**：`answerable`(78) + `must_refuse`(12) 上**必须零命中**；
* **召回（尽量高）**：`must_route_general`(17) 命中越多越好，但**不许为了召回牺牲精度**。

用法：`python scripts/check_route_rules.py [--qa-file eval/qa_set.jsonl]`
退出码：0 精度达标；1 有误伤（会把命中项逐条打印出来）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from legal_rag.route_intent import non_legal_reason  # noqa: E402

#: 这两类**绝不允许**被预路由拦下
PROTECTED = ("answerable", "must_refuse")


def main() -> int:
    parser = argparse.ArgumentParser(description="预路由规则的精度/召回实测")
    parser.add_argument("--qa-file", default=str(ROOT / "eval" / "qa_set.jsonl"))
    args = parser.parse_args()

    rows = [json.loads(line) for line in Path(args.qa_file).read_text(
        encoding="utf-8").splitlines() if line.strip()]

    hits = {row["id"]: non_legal_reason(str(row.get("question") or "")) for row in rows}
    fired = {item_id: reason for item_id, reason in hits.items() if reason}

    false_positive = [row for row in rows
                      if row["expect"] in PROTECTED and hits[str(row["id"])]]
    general = [row for row in rows if row["expect"] == "must_route_general"]
    caught = [row for row in general if hits[str(row["id"])]]

    print(f"评测集 {len(rows)} 题；规则命中 {len(fired)} 条")
    print(f"  精度：protected（answerable {sum(1 for r in rows if r['expect'] == 'answerable')}"
          f" + must_refuse {sum(1 for r in rows if r['expect'] == 'must_refuse')}）"
          f" 误伤 **{len(false_positive)}** 条（必须为 0）")
    for row in false_positive:
        print(f"    !! 误伤 {row['id']}（{row['expect']}）：{row['question']}"
              f"  ← {hits[str(row['id'])]}")
    print(f"  召回：must_route_general {len(caught)}/{len(general)} = "
          f"{len(caught) / max(len(general), 1):.0%}")
    for row in general:
        marker = "命中" if hits[str(row["id"])] else "漏"
        print(f"    {marker} {row['id']}: {row['question'][:40]}"
              f"{'  ← ' + hits[str(row['id'])] if hits[str(row['id'])] else ''}")

    return 1 if false_positive else 0


if __name__ == "__main__":
    raise SystemExit(main())
