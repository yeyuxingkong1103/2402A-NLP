# -*- coding: utf-8 -*-
"""把症状覆盖层导成人读的复核清单（与口语映射表的 build_colloquial_map.py 同一套路）。

用法：
    python scripts/make_symptoms_checklist.py
输出：
    data/processed/症状挂卡清单.md
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from validate_symptoms import collect  # noqa: E402  复用同一套加载口径

OUT = os.path.join(ROOT, "data", "processed", "症状挂卡清单.md")


def main():
    rows, errs, stats = collect()
    lines = ["# 症状挂卡清单（人工复核用，脚本现算）", "",
             f"- 条目数：**{stats['n']}**",
             f"- 按作物：{stats['by_crop']}",
             f"- 按来源：{stats['by_kind']}",
             f"- 校验：{'✗ ' + str(len(errs)) + ' 个错误' if errs else '✓ 通过'}", "",
             "> 逐条核对：症状原文是否与目标病虫害相符、出处是否可回查、"
             "是否与评估集问句同源（12396）。核对后把 `needs_human_review` 改成 false。", "",
             "| # | 作物 | 症状挂到 | 病虫害 | 症状原文 | 出处 | 复核 |",
             "| ---: | --- | --- | --- | --- | --- | --- |"]
    for i, r in enumerate(rows, 1):
        src = r.get("source") or {}
        ref = " ".join(x for x in (src.get("title"), src.get("publisher")) if x)
        txt = (r.get("symptom_text") or "").replace("\n", " ")
        flag = "⬜ 待复核" if r.get("needs_human_review") else "✅ 已复核"
        lines.append(f"| {i} | {r.get('crop')} | `{r.get('card_id')}` | {r.get('pest')} | "
                     f"{txt[:60]} | {ref} | {flag} |")
    if errs:
        lines += ["", "## 校验错误", ""] + [f"- {e}" for e in errs]
    lines += ["", "## 未覆盖清单", "",
              "> 本批计划要覆盖、但没落库的目标（找不到可引用出处就不硬凑，逐条写明原因）。", ""]
    covered = {r.get("card_id") for r in rows}
    plan_targets = [
        ("GBZ26583-2011-p6-b6.2", "辣椒 疫病（正文卡）"),
        ("GBZ26578-2011-p20-c09", "大蒜 豌豆潜叶蝇"),
        ("GBZ26578-2011-p20-c10", "大蒜 蓟马"),
        ("GBZ26583-2011-p21-c05", "辣椒 茶黄螨"),
        ("GBZ26583-2011-p20-c01", "辣椒 猝倒病、立枯病"),
    ]
    miss = [(cid, name) for cid, name in plan_targets if cid not in covered]
    lines += ([f"- [ ] `{cid}` {name}" for cid, name in miss] if miss
              else ["- （无，5 条全部覆盖）"])
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"→ {OUT}（{len(rows)} 条）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
