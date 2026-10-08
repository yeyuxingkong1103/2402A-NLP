"""保护类规则（FR-015 ~ FR-019）。

保护类不是"一条条去执行的规则"，而是一组**绝不可被剔除规则命中的判据**。
清洗前后各统计一次：某个判据在清洗前 >0、清洗后归零，说明规则误杀了，
必须非 0 退出告警——这正是 docs/04 §6 说的"最危险的一类缺陷"。
"""

from __future__ import annotations

import re

# 注册的保护判据（FR-018）。键即报告里显示的名字。
PROTECTED_PATTERNS = {
    # 宪法原则 II：强制溯源引用
    "citation:sup": re.compile(r"<sup>"),
    "citation:bracket": re.compile(r"\[\d+(?:[-,]\d+)*\]"),
    # 宪法原则 IV：紧急症状前置响应
    "emergency:triage": re.compile(
        r"转诊|急救|急诊|意识丧失|昏迷|抽搐|胸痛|呼吸困难|剧烈头痛|立即"
    ),
    # 剂量与数值：清洗不得改动其语义
    "dosage:unit": re.compile(
        r"\d+(?:\.\d+)?\s*(?:mmHg|mg|g|ml|mL|μg|℃|次/分|次/min)"
    ),
}

RECORD_PREFIX = "protection:"


def count_patterns(texts) -> dict[str, int]:
    """统计一组文本里各保护判据的命中数。"""
    joined = "\n".join(texts)
    return {name: len(rx.findall(joined)) for name, rx in PROTECTED_PATTERNS.items()}


def check_regression(before: dict[str, int], after: dict[str, int]) -> list[str]:
    """返回被误杀的判据清单（清洗前有、清洗后没了）。"""
    killed = []
    for name, n_before in before.items():
        if n_before > 0 and after.get(name, 0) == 0:
            killed.append("%s（清洗前 %d 处，清洗后 0 处）" % (name, n_before))
    return killed


def format_counts(counts: dict[str, int]) -> str:
    return "  ".join("%s=%d" % (k, v) for k, v in sorted(counts.items()))
