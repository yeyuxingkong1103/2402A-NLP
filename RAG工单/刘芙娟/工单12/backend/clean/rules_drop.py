"""剔除类规则（FR-009 ~ FR-014）。

每一条规则都有**具体名字**，写进 dropped.jsonl 的 drop_rule。
笼统的 "dropped" 会让误杀无法定位（docs/04 §6 明确禁止）。
"""

from __future__ import annotations

from dataclasses import dataclass

# 按类型直接剔除（docs/04 §6 处置表）
DROP_TYPE_RULES = {
    "page_number": "type:page_number",   # 印刷页码，物理页码另由 page_idx 计算
    "header": "type:header",             # 页眉：重复书名/刊名，造成跨页假重复
    "footer": "type:footer",             # 页脚
    "aside_text": "type:aside_text",     # 侧边装饰文字
}

# 注册的规则全集：用于在报告里区分「命中 0 条」与「未实现」（FR-035）
REGISTERED_RULES = (
    "type:page_number",
    "type:header",
    "type:footer",
    "type:aside_text",
    "list:ref_text",
    "empty:text",
    "opt:page_footnote",
)


@dataclass
class DropDecision:
    rule: str
    text: str
    char_len: int


def decide(block: dict, rendered_text: str, opts) -> DropDecision | None:
    """返回 None 表示保留；否则返回命中规则与用于留痕的文本。"""
    btype = block.get("type", "")
    text = rendered_text or ""
    char_len = len(text)

    # 1) 版式噪声类型
    rule = DROP_TYPE_RULES.get(btype)
    if rule:
        return DropDecision(rule, text, char_len)

    # 2) 参考文献条目（面向群众的问答不引用文献条目）
    if btype == "list" and block.get("sub_type") == "ref_text":
        return DropDecision("list:ref_text", text, char_len)

    # 3) 页脚注释：默认保留（医疗指南页脚常是剂量/注意事项注解，
    #    误杀代价高于噪声代价）。只有显式开关才丢。
    if btype == "page_footnote" and getattr(opts, "drop_page_footnote", False):
        return DropDecision("opt:page_footnote", text, char_len)

    # 4) 空块
    if not text.strip():
        return DropDecision("empty:text", text, char_len)

    return None
