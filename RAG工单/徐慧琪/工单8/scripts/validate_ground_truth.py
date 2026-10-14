# 工单编号：人工智能NLP-RAG-基于GraphRAG实现金融问答
"""
金标准质量闸门（评估前必跑）。

逐条统计「检索判定事实集」中每个 fact 在**全语料**中的命中块数：
  · 0 命中   → 该事实不可能被检索到，金标准写错或证据只在图像中；
  · 命中过泛 → 命中 > 200 块，无区分度，指标会被稀释。
两类问题按设计都会让评估结果失真，必须在评估前修掉，脚本以退出码 1 报警。

检索判定事实集怎么取（与工单07 的 validate_ground_truth.py 同口径）
--------------------------------------------------------------------
07 的金标准 schema 把两类事实分开（沿用至今）：
  · `evidence_facts` —— **检索命中判定集**：判「这个块是否相关」用它，
    因此它必须是文本层里逐字可匹配的原文串（召回分母/相关性都靠它）；
  · `key_facts`      —— **答案评分集**：判「答案说对了没有」用它，
    允许包含只看得到答案才需要的词（如 07 的图像题 Q10：板块名/部门名
    只存在于 PDF 内嵌图像里，文本层 0 命中是**设计预期**，不是缺陷）。

本脚本按**来源（origin）**而非「字段是否存在」决定闸门口径：
  · `origin == "wo07"`（Q01~Q10，冻结、不得改动）：沿用 07 口径，
    `evidence_facts` 优先、缺省回退 `key_facts`；未被闸门覆盖的 key_facts
    条目以「提示」列出（不阻塞，如 Q10 图像题的答案词只存在于 PDF 图像层）；
  · 其余（`origin == "wo08"`，含未来新增题）：`key_facts` **一律**是致命的
    检索判定集，即使携带 `evidence_facts` 字段也不豁免——任一事实 0 命中 /
    命中 > 200 直接退出码 1，防止用两字段 schema 悄悄绕过闸门。

检索判定事实集为空（无任何条目可查）同样按金标准问题报警，不静默放过。

答案评分集（key_facts）中不参与闸门、但命中 0 / 过泛的 wo07 条目，会在末尾以
「提示」列出（不阻塞，供人工确认是否为设计预期）。

用法:
    python scripts/validate_ground_truth.py
"""
from __future__ import annotations

import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:            # 支持 `python scripts/validate_ground_truth.py`
    sys.path.insert(0, ROOT)

from src import bootstrap  # noqa: F401,E402  —— 必须最先导入（离线环境 + 栈修复）
from src import config  # noqa: E402

OVERBROAD = 200


def _load_chunks() -> list[dict]:
    path = os.path.join(config.CHUNK_DIR, "chunks.json")
    with io.open(path, encoding="utf-8") as f:
        return json.load(f)


def gate_facts(item: dict) -> tuple[list[str], str]:
    """返回 (检索判定事实集, 来源字段名)。

    口径按 origin 定，而不是看 evidence_facts 字段在不在：
      · wo07（冻结的 Q01~Q10）：evidence_facts 优先、缺省回退 key_facts；
      · 其余（wo08 及其后续新增题）：一律取 key_facts（始终致命），
        带不带 evidence_facts 都不降级，避免两字段 schema 绕过闸门。
    """
    if item.get("origin") == "wo07":
        ev = item.get("evidence_facts")
        if ev:
            return list(ev), "evidence_facts"
    return list(item.get("key_facts") or []), "key_facts"


def main() -> int:
    chunks = _load_chunks()
    with io.open(os.path.join("data", "eval", "ground_truth.json"), encoding="utf-8") as f:
        gt = json.load(f)

    texts = [(c.get("text", "") or "") + "\n" + (c.get("parent_text", "") or "")
             for c in chunks]

    problems: list[str] = []
    notes: list[str] = []

    print(f"语料块 {len(chunks)}；金标题 {len(gt)} 道")
    print(f"\n{'题':>4} {'事实（检索判定集）':<34} {'命中块数':>8}  状态")
    print("-" * 72)
    for it in sorted(gt, key=lambda x: x["id"]):
        facts, field = gate_facts(it)
        if not facts:
            problems.append(f"Q{it['id']}: 检索判定事实集为空（{field} 无任何条目）")
        for fact in facts:
            n = sum(1 for t in texts if fact in t)
            if n == 0:
                status = "× 零命中"
                problems.append(f"Q{it['id']}: 「{fact}」全语料 0 命中")
            elif n > OVERBROAD:
                status = "! 过泛"
                problems.append(f"Q{it['id']}: 「{fact}」命中 {n} 块，过泛")
            else:
                status = "√"
            print(f"{it['id']:>4} {fact[:34]:<34} {n:>8}  {status}")
        # wo07 答案评分集里与检索集不同的条目：0 命中/过泛只提示，不阻塞
        # （wo08 的 key_facts 本身就是闸门事实集，不进入此豁免分支）
        if it.get("origin") == "wo07" and field == "evidence_facts":
            for fact in it.get("key_facts") or []:
                if fact in facts:
                    continue
                n = sum(1 for t in texts if fact in t)
                if n == 0:
                    notes.append(f"Q{it['id']}: 答案评分集「{fact}」0 命中"
                                 "（07 原样保留：图像题证据在 PDF 图像层）")
                elif n > OVERBROAD:
                    notes.append(f"Q{it['id']}: 答案评分集「{fact}」命中 {n} 块，过泛"
                                 "（07 原样保留：仅用于答案评分，不参与检索判定）")

    print("-" * 72)
    if notes:
        print("提示（不阻塞；07 的 Q01~Q10 原样沿用，不修改）：")
        for nt in notes:
            print("  -", nt)
        print()
    if problems:
        print(f"发现 {len(problems)} 个金标准问题：")
        for p in problems:
            print("  -", p)
        print("\n请在评估前修正（可拆分为 evidence_facts / key_facts，"
              "或移出无区分度的事实）。")
        return 1
    print("√ 金标准自检通过：检索判定集无零命中、无过泛。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
