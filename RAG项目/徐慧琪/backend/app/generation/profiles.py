"""两侧提示词。

存在的理由：律师侧与公众侧的差异（FR-7.4 vs FR-8.3~8.6）不能靠"同一段
提示词碰运气"——公众侧禁止给个案结论是合规要求，必须写死在这里。

提示词正文集中在本模块，不放 chain.py：LCEL 链要保持薄，而提示词是最常改的
东西，混在链里会让每次改字都碰到编排代码（技术方案 2.2 摩擦点 4）。
"""
from __future__ import annotations

SIDE_INTERNAL = "internal"
SIDE_PUBLIC = "public"

# 老问题：3b 模型的 JSON 合规靠提示压。共同约束抽出来，两侧只写差异
_COMMON_RULES = """你是法律条文检索助手。

硬性要求：
1. 只能依据【法条原文】作答，不得引用未提供的条文，严禁编造条号或法条内容。
2. 每条结论都要给出引用；引用里的 quote 必须是【法条原文】中逐字出现的片段，
   长度不少于 10 个字，不得改写、不得拼接。
3. 若【法条原文】不足以回答，输出 status = "need_more_info" 并在 answer 里
   说明还需要哪些信息（如时间、地点、身份、证据）。
4. 若问题不是法律问题、或属于紧急求助，输出 status = "out_of_scope"。
5. 输出必须是 JSON，字段为 status / answer / citations / disclaimer。
   citations 是数组，每项含 law / article / paragraph / item / quote；
   整条引用时 paragraph 与 item 留空。"""

_INTERNAL_ONLY = """
6. 面向上班族律师：answer 用规范表述，可给出可直接复制进文书的引用格式
   （如「《中华人民共和国民法典》第五百八十四条」）。"""

_PUBLIC_ONLY = """
6. 面向普通公众：用通俗的语言解释，不使用法言法语。
7. 不得给出个案结论、不得预测胜诉与否、不得代替律师出具意见；
   涉及具体纠纷时只说明法律规定的通常情形。
8. 必须提醒本回答仅供参考、不构成正式法律意见、建议咨询执业律师。"""

_PROMPTS = {
    SIDE_INTERNAL: _COMMON_RULES + _INTERNAL_ONLY,
    SIDE_PUBLIC: _COMMON_RULES + _PUBLIC_ONLY,
}


def system_prompt(side: str) -> str:
    """按侧取系统提示词。未知侧直接抛错，不静默退回律师侧提示词。"""
    if side not in _PROMPTS:
        raise ValueError(f"未知的侧别：{side!r}，只支持 {sorted(_PROMPTS)}")
    return _PROMPTS[side]


def build_payload(question: str, blocks: list[dict],
                  history: list[str] | None = None) -> str:
    """拼 human 消息：法条原文 + 会话上下文 + 被定界的用户问题。

    定界符 <<< >>> 是防提示注入（技术方案 9.4）——用户文本里出现
    "忽略以上指令"时，模型至少能看出它不是系统给的。
    """
    if blocks:
        lines = []
        for index, block in enumerate(blocks, start=1):
            # 标出条号与路径：模型据此写 citation，不给就等于让它自己编
            lines.append(f"{index}) 《中华人民共和国民法典》第{block['article_no']}条"
                         f"（{block['path']}）：{block['text']}")
        articles = "\n".join(lines)
    else:
        articles = "（未检索到相关法条）"
    context = ""
    if history:
        context = "\n\n【会话上下文】\n" + "\n".join(history)
    return (f"【法条原文】\n{articles}{context}\n\n"
            f"【用户问题】\n<<<\n{question}\n>>>")
