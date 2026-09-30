#!/usr/bin/env python3
"""从评测集里**派生扰动探针集**：同一批分流题换个说法再问一遍。

为什么需要它（2026-09-28）：六臂那轮里 v5d 的分流子集 26/29（基座 24/29）看着超标尺，
但**原样重跑证明不了任何东西** —— 同条件两窗口的臂读数逐位相同（解码确定性）。
要检验"这个优势是真的还是噪声"，只能**扰动**：同样的问题换措辞/加诱导，看优势是否保持。

三种扰动（都是**机械、可复现、不调模型**的）：

* ``p1_polite``     礼貌前缀（"请问，"）—— 最轻的表面变化；
* ``p2_surface``    同义替换表（是不是→算不算 等）+ 兜底后缀 —— 中等表面变化；
* ``p3_legal_bait`` "根据法律规定，"前缀 —— **诱导**：看模型会不会被带进法律模式
  （这正是 §D7 里"口算题被判成法律问题"的同一类失效）。

⚠️ **它不是评测集的一部分**：`eval/qa_set.jsonl` 的 107 题保持冻结；
本文件只从其中的 **29 道分流题**（must_refuse + must_route_general）派生，
每条带 ``probe`` / ``derived_from`` / ``perturbation`` 字段，便于追溯。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

#: 同义替换表：**只收语义等价的改写**，拿不准就不收（宁可退回后缀兜底）
SURFACE_RULES: tuple[tuple[str, str], ...] = (
    ("是不是", "算不算"),
    ("怎么办", "该怎么处理"),
    ("帮我算一下", "帮我算算"),
    ("可以吗", "行不行"),
    ("能不能", "可不可以"),
    ("如何", "怎么"),
    ("哪些", "有什么"),
    ("是否", "算不算"),
    ("属于", "算不算"),
    ("麻烦", "劳烦"),
)

#: p2 没命中任何规则时的兜底后缀（保证与原文不同，且不改变意图）
SURFACE_FALLBACK_SUFFIX = "，麻烦说得具体一点。"

#: 分流题（这两种 expect 才看"分流行为"）
ROUTE_EXPECT = ("must_refuse", "must_route_general")


def perturb(question: str) -> list[tuple[str, str, list[str]]]:
    """返回 ``[(扰动名, 扰动后的问题, 命中的规则)]``。纯函数，便于单测。"""
    text = question.strip()
    variants: list[tuple[str, str, list[str]]] = []

    polite = f"请问，{text}"
    variants.append(("p1_polite", polite, []))

    swapped = text
    fired: list[str] = []
    for source, target in SURFACE_RULES:
        if source in swapped:
            swapped = swapped.replace(source, target, 1)
            fired.append(f"{source}->{target}")
    if swapped == text:
        swapped = f"{text.rstrip('。？！?')}{SURFACE_FALLBACK_SUFFIX}"
        fired.append("fallback_suffix")
    variants.append(("p2_surface", swapped, fired))

    variants.append(("p3_legal_bait", f"根据法律规定，{text}", []))
    return variants


def build(rows: list[dict], *, probe_arms: tuple[str, ...] = ("p1_polite", "p2_surface",
                                                              "p3_legal_bait")) -> list[dict]:
    out: list[dict] = []
    for row in rows:
        if str(row.get("expect")) not in ROUTE_EXPECT:
            continue
        for name, question, fired in perturb(str(row.get("question") or "")):
            if name not in probe_arms:
                continue
            item = dict(row)
            item["id"] = f"{row.get('id')}~{name}"
            item["question"] = question
            item["probe"] = "perturbation"
            item["derived_from"] = str(row.get("id"))
            item["perturbation"] = name
            item["perturbation_rules"] = fired
            out.append(item)
    return out


def load_qa(path: Path) -> list[dict]:
    handle = path.open(encoding="utf-8")
    with handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="派生扰动探针集（不修改原评测集）")
    parser.add_argument("--qa-file", default="eval/qa_set.jsonl")
    parser.add_argument("--out", default="eval/perturb-routing.jsonl")
    args = parser.parse_args()

    rows = load_qa(Path(args.qa_file))
    items = build(rows)
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    by_kind: dict[str, int] = {}
    for item in items:
        by_kind[item["perturbation"]] = by_kind.get(item["perturbation"], 0) + 1
    print(f"原集 {len(rows)} 题，派生扰动探针 {len(items)} 条 -> {target}")
    print(f"  按扰动类型：{by_kind}")
    print(f"  来源：must_refuse + must_route_general 共 "
          f"{sum(1 for row in rows if str(row.get('expect')) in ROUTE_EXPECT)} 题")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
