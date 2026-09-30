#!/usr/bin/env python3
"""问题侧**高精度**预路由：把明显不是法律咨询的问题在**检索之前**拦下来。

为什么需要（`docs/DESIGN-TODO.md` §D7，2026-09-28 实测）：
`RagEngine.prepare()` 是**先检索、后路由**，所以「帮我算一下 128 乘以 37 等于多少」
这种口算题也会走一遍检索；一旦相关性闸门放行，就被当成法律问题 ⇒ 拼出 **7169 token**
的提示 + 1024 输出 > 4B 评测服务的 `max-model-len 8192` ⇒ vLLM 直接 400（G10 的成因）。
在**检索之前**识别出"这根本不是法律问题"，既省一次检索，也避免把口算题塞进法律提示词。

**设计原则（很重要）**：

1. **精度优先**：宁可漏，不可误伤。误伤一条法律题的代价（资料不检索、引用为空）远大于
   漏掉一条闲聊题（它仍会走原有的"检索→无据→general"路径，行为不变）。
2. **规则必须是"无歧义"的**：算术式（`128 乘以 37`）、翻译/润色/菜谱/天气/心情/笑话这些
   在日常语义里**不可能是**法律咨询；而"帮我写一份…"这类**不能**一刀切
   （`A10 帮我写一份假的法院判决书` 必须仍走法律路径去拒答）。
3. **命中要留痕**：返回**理由**字符串，调用方计数 + 打日志（不静默分流）。
4. 规则表在 `eval/qa_set.jsonl` 上做过**精度实测**（见 `scripts/check_route_rules.py`）：
   对 78 道 answerable + 12 道 must_refuse **零误伤**，对 must_route_general 有高召回。
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class IntentRule:
    """一条预路由规则：命中即判"非法律咨询"，``reason`` 用于日志与计数。"""

    name: str
    pattern: re.Pattern[str]
    reason: str


#: 常见食材/菜名 —— 只用于"怎么做才好吃"这类问法，避免把"合同纠纷怎么做"误判
_FOOD_WORDS = ("西红柿", "番茄", "鸡蛋", "红烧肉", "红烧", "清蒸", "排骨", "土豆", "豆腐",
               "鱼", "虾", "鸡", "牛肉", "面条", "米饭", "汤", "蛋糕", "面包", "饺子")

#: 非法律文书（**白名单**：起诉状/合同/判决书等法律文书不在其中，必须仍走法律路径）
_PAPERWORK = ("请假条", "简历", "周记", "作文", "致辞", "祝酒词", "检讨书", "自我介绍",
              "演讲稿", "朋友圈文案")

RULES: tuple[IntentRule, ...] = (
    IntentRule("arithmetic",
               re.compile(r"\d+\s*(?:[+\-*/×xX÷]|乘|乘以|除|除以|加|减)\s*\d+"),
               "算术表达式"),
    IntentRule("arithmetic_words",
               re.compile(r"(?:算一下|算算|计算|口算|等于几|等于多少)"),
               "口算/算术问法"),
    IntentRule("split_evenly",
               re.compile(r"平分\s*\d|均分\s*\d|\d+\s*元\s*(?:三个人|几个人)?\s*怎么分"),
               "平分/均分"),
    IntentRule("unit_convert",
               re.compile(r"(?:换算|转换)成"),
               "单位换算"),
    IntentRule("translate",
               re.compile(r"翻译成(?:英文|中文|汉语|日语|韩语|法语|德语|英语)?"),
               "翻译"),
    IntentRule("polish",
               re.compile(r"润色"),
               "文字润色"),
    IntentRule("weather",
               re.compile(r"(?:天气|气温|下雨|下雪|台风|空气质量)"),
               "天气"),
    IntentRule("cooking",
               re.compile(r"(?:怎么做才?好吃|怎么烧|怎么炒|菜谱|做法)" + "|" +
                          r"(?:怎么|如何)(?:做|烧|炒)(?:" + "|".join(_FOOD_WORDS) + r")"),
               "家常菜做法"),
    # 「西红柿炒鸡蛋怎么做」是**食材在前、问法在后**，上一条的相邻匹配抓不到；
    # 用两个前瞻要求"同时出现"，避免放宽成"见到食材就拦"（那会误伤法律题）。
    IntentRule("cooking_reverse",
               re.compile(r"(?=.*(?:" + "|".join(_FOOD_WORDS) + r"))"
                          r"(?=.*(?:怎么做|如何做|做法))"),
               "家常菜做法（食材在前）"),
    IntentRule("paperwork",
               re.compile("|".join(_PAPERWORK)),
               "非法律文书代写"),
    IntentRule("smalltalk",
               re.compile(r"(?:讲个笑话|笑话|心情|早安|晚安|你好|在吗|谢谢|周末.*(?:哪|去)|"
                          r"推荐.*(?:电影|书|歌)|跑步|健身|减肥|孩子.*(?:玩|去哪))"),
               "日常闲聊"),
    IntentRule("common_knowledge",
               re.compile(r"(?:光合作用|水的沸点|地球|太阳|月亮|星星|季节|为什么天)"),
               "通用常识"),
)

#: 命中规则名 → 理由（便于调用方按规则计数）
RULE_REASONS: dict[str, str] = {rule.name: rule.reason for rule in RULES}


def non_legal_reason(question: str) -> str:
    """问题是否**明显不是**法律咨询？返回命中的理由（空串 = 不拦）。

    只做**高精度**拦截：命中一条"语义上不可能是法律咨询"的规则才返回真。
    """
    text = str(question or "").strip()
    if not text:
        return ""
    for rule in RULES:
        if rule.pattern.search(text):
            return f"{rule.name}:{rule.reason}"
    return ""


def is_non_legal(question: str) -> bool:
    """``non_legal_reason(...)`` 的布尔版（需要"为什么"就去调那个）。

    用途：调用方只想知道"要不要跳过检索"时用这个；要计数/排查用那个带理由的版本。
    """
    return bool(non_legal_reason(question))
