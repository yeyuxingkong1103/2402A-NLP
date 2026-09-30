"""输出边界：拒答、免责注入。

存在的理由（FR-8.3、FR-8.4、AC-18、AC-19）：三件事都是合规项，合规项不能
依赖模型的自觉。拒答用正则规则（确定性、可穷举测试），免责由代码注入并
覆盖模型输出——模型漏写或改写都不影响结果。

与提示词的分工（profiles.py 公众侧第 7、8 条已写明同样要求）：那边是"让模型
尽量做对"，这里是"模型没照做时仍然成立"的兜底。于是本文必须是纯代码、无模型、
无 IO，且只依赖 schema 与 profiles——不碰检索层，也不给编排层加接口。
"""
from __future__ import annotations

import re

from app.generation.profiles import SIDE_INTERNAL, SIDE_PUBLIC
from app.generation.schema import Answer

# 拒答话术。拒绝之后必须给出路——只说不答会让用户直接流失
REFUSAL_REPLY = ("这个问题需要结合你的具体情况判断，我不能替你做结论。"
                 "建议携带材料咨询执业律师；我可以先帮你查相关的法律规定。")

# 公众侧固定免责声明（FR-8.4、AC-18）
PUBLIC_DISCLAIMER = ("本回答仅供参考，不构成正式法律意见，也不能替代律师的专业判断；"
                     "具体纠纷请咨询执业律师。")

# 拒答规则。每条都是"明确在要个案结论 / 要代理服务"的句式，
# 刻意写窄：宁可漏拒也不要误伤正常提问。
REFUSAL_PATTERNS = [
    # 胜诉预测类
    re.compile(r"(能不能|能否|会不会|可以|能)(赢|胜诉|打赢|输)"),
    re.compile(r"(胜诉|败诉|打赢|输赢)(的)?(概率|几率|可能性|把握)"),
    re.compile(r"这个?案子(我)?(能|会)(不能|不会)?(赢|输)"),
    # 代写文书类
    re.compile(r"(帮我|替我|给我)(写|起草|拟)(一份|个|一份儿)?(起诉状|答辩状|上诉状|合同|协议|律师函)"),
    # 代理服务类
    re.compile(r"(帮我|替我)(代理|打官司|出庭)"),
    re.compile(r"(你|能不能)(代理|接)(我|这个)的?(案子|case)"),
]


def is_refusal(question: str) -> bool:
    """命中任一拒答规则即拒答。纯规则、无模型参与。"""
    text = (question or "").strip()
    return any(pattern.search(text) for pattern in REFUSAL_PATTERNS)


def apply_boundaries(ans: Answer, side: str) -> Answer:
    """按侧补正免责声明，返回**新对象**（不改入参，调用方可能还要用原值）。

    只动 disclaimer：status 的三个合法值由 schema 的 Literal 钉死，
    error/abstain 是编排层的事，这里越权改 status 会让调用方拿到自相矛盾的答案。

    未知侧抛错而非静默当作内部侧：与 profiles.system_prompt、llm_router.get_llm
    同一口径。这是合规兜底模块，侧别写错时最危险的失败方向是"公众答案没有免责"，
    所以宁可当场炸掉也不能 fail-open。
    """
    if side not in (SIDE_INTERNAL, SIDE_PUBLIC):
        raise ValueError(f"未知的侧别：{side!r}")
    disclaimer = PUBLIC_DISCLAIMER if side == SIDE_PUBLIC else ""
    return ans.model_copy(update={"disclaimer": disclaimer})
