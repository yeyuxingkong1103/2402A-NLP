"""按四类（剔除 / 保护 / 转换 / 修复）分组的处置报告（FR-034、FR-035）。

关键约束：某类命中 0 条与「该类未实现」必须可区分。做法是先把全部规则名
**注册**进来，报告时以注册表为准、以计数为 0 显示。
"""

from __future__ import annotations

from . import rules_drop, rules_protect

CATEGORIES = ("drop", "protect", "convert", "repair")

CATEGORY_LABEL = {
    "drop": "剔除类",
    "protect": "保护类",
    "convert": "转换类",
    "repair": "修复类",
}

REGISTERED = {
    "drop": rules_drop.REGISTERED_RULES,
    "protect": tuple(
        ["%s%s" % (rules_protect.RECORD_PREFIX, k)
         for k in rules_protect.PROTECTED_PATTERNS]
        + ["protection:citation_count_stable", "protection:dropped_content_risk"]
    ),
    "convert": (
        "convert:backslash_unescape",   # \\~ -> ~
        "convert:html_entity",          # &lt; -> <
        "convert:table_to_markdown",    # 表体 HTML -> Markdown
        "convert:cjk_space_merge",      # D1 词内空格合并
        "convert:page_number",          # page_idx -> 1-based page
        "convert:heading_detect",       # text_level 键存在即标题
    ),
    "repair": (
        "repair:cross_page_merge",          # 2-C 跨页断句合并
        "repair:split_glued_heading",       # 2-B 粘连标题切出还原
        "repair:retag_untagged_heading",    # 3-D 补标漏标标题
        "repair:demote_caption_heading",    # 3-D 题注误标降级
        "repair:recompute_heading_level",   # 3-D 按编号重算层级
    ),
}


class Report:
    def __init__(self) -> None:
        self.counts: dict[str, dict[str, int]] = {c: {} for c in CATEGORIES}
        self.notes: dict[str, list[str]] = {c: [] for c in CATEGORIES}
        self.samples: dict[str, list[str]] = {c: [] for c in CATEGORIES}

    def hit(self, category: str, rule: str, n: int = 1) -> None:
        bucket = self.counts[category]
        bucket[rule] = bucket.get(rule, 0) + n

    def note(self, category: str, message: str, sample: str = "") -> None:
        if message not in self.notes[category]:
            self.notes[category].append(message)
        if sample and len(self.samples[category]) < 12:
            self.samples[category].append(sample)

    def total(self, category: str) -> int:
        return sum(self.counts[category].values())

    def render(self) -> str:
        lines = ["", "=" * 72, "清洗处置报告（按四类）", "=" * 72]
        for category in CATEGORIES:
            label = CATEGORY_LABEL[category]
            lines.append("")
            lines.append("【%s】命中合计 %d" % (label, self.total(category)))
            registered = REGISTERED[category]
            if not registered:
                lines.append("  （该类未实现）")
                continue
            for rule in registered:
                n = self.counts[category].get(rule, 0)
                lines.append("  %-32s %6d 条" % (rule, n))
            extra = [r for r in self.counts[category] if r not in registered]
            for rule in sorted(extra):
                lines.append("  %-32s %6d 条  (未注册)" % (rule, self.counts[category][rule]))
            for message in self.notes[category]:
                lines.append("  [提示] %s" % message)
            for sample in self.samples[category][:5]:
                lines.append("      · %s" % sample)
        lines.append("")
        return "\n".join(lines)
