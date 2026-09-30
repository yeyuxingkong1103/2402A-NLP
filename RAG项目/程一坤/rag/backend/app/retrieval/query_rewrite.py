"""基于短期对话上下文的查询改写。

要解决的问题：用户追问常说"那试用期呢？"——单独拿去检索，
向量/BM25 都不知道"那"指什么。本模块在检索前把指代/省略补全成
"经济补偿 12 期"这类自包含查询（上一轮原文 + 本轮剩余部分，不再补"怎么算"，见 _compose_query），
只做规则改写（不调 LLM），失败一律回退原问题，保证检索链路永不被改写环节拖垮。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

# 显式指代前缀：以这些词开头的追问几乎必然依赖上文
_REFERENCE_PREFIXES = ("那", "这个", "这个问题", "上述", "前面")
# 疑问后缀：_extract_topic 用它剥掉主题词尾部的疑问词（"经济补偿怎么算" → "经济补偿"）。
# 注意别再往里加 "？"/"?"：上一步已把标点替换成空格，标点类后缀永远判不中（批次 24 删除死项）。
_QUESTION_SUFFIXES = ("怎么回事", "怎么算", "是什么", "有哪些", "如何", "吗", "呢")
# 单字碎片首字：这些字不能单独成词（"个""的""些"），只能粘在别的字后面用（"那个""是的""这些"）。
# 剥指代前缀时若把它们留在剩余串开头（"那个员工" → "个员工"），说明这一刀把词切碎了，
# 生成的是"主题 个员工的工资怎么算"这种脏查询 —— 遇到就放弃这一刀（批次 24b 引入）。
# 批次 26-C 扩充到"量词/助词单字"：单个量词（些/种/位/名/只/条）留在查询开头同样是碎片
# （"那些情况怎么算" → "些情况怎么算"）。
# 注：原先还列了"那""这"，本批按裁决换成上面这一组——核查过它们不承担拦截职责：
#   · 剩余串以"那"开头时必然也以指代前缀"那"开头，走下面的例外分支放过，不靠这一项拦
#   · 剩余串以"这"开头但不是"这个/这个问题"时（"这些呢"），拦下来只会退回"主题 那这些呢"，
#     比放过它得到的"主题 这些"更差
_FRAGMENT_LEAD_CHARS = ("个", "的", "些", "种", "位", "名", "只", "条")


def _strip_question_noise(text: str) -> str:
    """去掉问题串首尾的标点与语气词"呢"，返回其中的实质内容。

    与 _compose_query 收尾清理是同一套规则，抽出来是为了在剥指代的循环里
    也能判断"这一刀剥完还剩不剩实质内容"：
    - 剩空了 ⇒ 整问都是指代（"那这个呢？"），收尾会返回主题词
    - 只剩 1 个字 ⇒ 碎片，放弃这一刀
    """
    text = re.sub(r"^[，,、\s]+", "", text)
    text = re.sub(r"[。！？?；;]+$", "", text).strip()
    return re.sub(r"呢$", "", text).strip()


@dataclass(frozen=True)
class QueryRewriteResult:
    """查询改写结果及可解释依据。

    Args:
        original_query: 用户原始输入（回退时也用它）
        rewritten_query: 改写后的查询；未改写时与 original_query 相同
        changed: 是否发生了改写（调用方据此决定是否留痕日志）
        reasons: 触发改写的依据清单（命中模式、主题词），供验收观测
    """

    original_query: str
    rewritten_query: str
    changed: bool
    reasons: list[str]


class QueryRewriter:
    """从最近有效的用户消息提取主题，补全当前问题中的指代。"""

    def rewrite(self, question: str, messages: list[Any] | None) -> QueryRewriteResult:
        """改写带有指代或省略的追问，失败时原样返回。

        Args:
            question: 当前用户问题
            messages: 短期记忆里的历史消息（role/content 字典列表），可为 None

        Returns:
            QueryRewriteResult：changed=True 表示已补全；任何前置条件不满足
            或改写结果反而变差时，都返回未改写结果（宁可不改，不可改错）
        """
        original_query = question
        # 防御：非字符串/空白输入直接原样返回，不抛异常
        if not isinstance(question, str) or not question.strip():
            return self._unchanged(original_query)

        # 前置闸门：没有指代信号（新问题）就没必要翻历史，省掉无谓的主题提取
        if not self._has_reference_signal(question):
            return self._unchanged(original_query)

        # 上文里找不到可用主题时无法补全，只能保持原样
        topic = self._latest_user_topic(messages)
        if not topic:
            return self._unchanged(original_query)

        rewritten_query = self._compose_query(topic, question)
        # 改写后与原问相同或为空，说明改写无增益，回退
        if not rewritten_query or rewritten_query == question:
            return self._unchanged(original_query)

        reasons = ["命中指代/省略模式"]
        reasons.append(self._matched_prefix(question))
        reasons.append("主题词：" + topic)
        return QueryRewriteResult(
            original_query=original_query,
            rewritten_query=rewritten_query,
            changed=True,
            reasons=reasons,
        )

    @staticmethod
    def _unchanged(question: str) -> QueryRewriteResult:
        """构造"未改写"结果（changed=False、无依据），统一回退出口。"""
        return QueryRewriteResult(
            original_query=question,
            rewritten_query=question,
            changed=False,
            reasons=[],
        )

    @staticmethod
    def _has_reference_signal(question: str) -> bool:
        """判定问题是否带指代信号。

        Args:
            question: 当前用户问题

        Returns:
            True 表示疑似追问：要么以显式前缀开头，要么匹配
            "这/那/其(+量词/条款项)(+疑问词)？结尾"的隐含指代正则
        """
        normalized = question.strip()
        # 正则兜住"这个呢？""那如何？"这类不以前缀开头、但明显指代上文的短问句
        return normalized.startswith(_REFERENCE_PREFIXES) or bool(
            re.search(r"(?:这|那|其)(?:个|种|项|期|条|款|部分)?(?:呢|吗|如何|怎么|多少)?[？?。]?$", normalized)
        )

    @staticmethod
    def _matched_prefix(question: str) -> str:
        """返回命中的指代前缀（用于可解释依据）；都没命中时报"隐含指代"。"""
        normalized = question.strip()
        # 按长度降序匹配：保证"这个问题"先于"这个"命中，依据才准确
        for prefix in sorted(_REFERENCE_PREFIXES, key=len, reverse=True):
            if normalized.startswith(prefix):
                return prefix
        return "隐含指代"

    @classmethod
    def _latest_user_topic(cls, messages: list[Any] | None) -> str | None:
        """从历史消息里倒序找最近一条能提取出主题的用户消息。

        Args:
            messages: 历史消息列表，元素预期是 {role, content} 字典

        Returns:
            主题词字符串；消息不足/格式异常/提取为空时返回 None
        """
        # 不是列表直接放弃（短期记忆缺失时可能是 None）
        if not isinstance(messages, list):
            return None
        # 少于 2 条说明当前问题基本就是首问，谈不上"追问"
        if len(messages) < 2:
            return None
        # 倒序遍历：取"最近"的用户消息当主题——离当前问题越近，指代越可能指向它
        for message in reversed(messages):
            if not isinstance(message, dict):
                continue
            if message.get("role") != "user":
                continue
            content = message.get("content")
            if not isinstance(content, str) or not content.strip():
                continue
            topic = cls._extract_topic(content)
            if topic:
                return topic
        return None

    @staticmethod
    def _extract_topic(content: str) -> str | None:
        """从一条用户消息中提炼主题词。

        Args:
            content: 用户消息正文

        Returns:
            清洗后的主题词；清洗后为空则返回 None
        """
        topic = content.strip()
        # 标点全部换成空格，避免主题词里带着问号影响后续检索分词
        topic = re.sub(r"[，。！？?；;：:、,.!]+", " ", topic)
        # 去掉疑问后缀（如"经济补偿怎么算"→"经济补偿"），主题只要名词核心
        for suffix in _QUESTION_SUFFIXES:
            if topic.endswith(suffix):
                topic = topic[: -len(suffix)].strip()
                break
        # 去掉客套语开头（"请问/我想问一下"），它们不是主题的一部分
        topic = re.sub(r"^(请问|请|我想问一下|想问一下)\s*", "", topic)
        topic = re.sub(r"\s+", " ", topic).strip()
        return topic or None

    @staticmethod
    def _compose_query(topic: str, question: str) -> str:
        """把主题词与追问剩余部分拼成自包含查询。

        Args:
            topic: 上一轮提取的主题词
            question: 当前追问原文

        Returns:
            拼接后的完整查询；追问除指代词外没有实际内容时，直接返回主题词
        """
        remainder = question.strip()
        # 剥掉开头指代词（长前缀优先，防止"这个问题"被"这个"剥一半）。
        # 循环剥到不再命中：只剥一层时"那这个呢？"会残留"这个"，
        # 下面"剥完指代后什么都不剩"那条分支形同虚设。
        # 批次 24 A/B（10 followup + 5 multi_turn，改前/改后各跑一遍）：
        # MRR@10 0.6444 → 0.6444 不变、第 2 轮 top5 与逐题排名一致；
        # 仅 2 道 multi_turn 第 2 轮改写串变化（"这个期间工资…"→"期间工资…"、
        # "这个最长能约定几年"→"最长能约定几年"），无一条变差 —— 故落地。
        prefixes = sorted(_REFERENCE_PREFIXES, key=len, reverse=True)
        while True:
            hit = next((p for p in prefixes if remainder.startswith(p)), None)
            if hit is None:
                break
            candidate = remainder[len(hit) :].strip()
            # 剥完还剩多少"实质内容"（去掉首尾标点与语气词"呢"后再看）
            payload = _strip_question_noise(candidate)
            if not payload:
                # 剥完只剩标点/语气词，说明整问都是指代（"那这个呢？"）：
                # 交给收尾的"空剩余 → 返回主题"分支，别当成碎片拦下来
                remainder = ""
                break
            # 兜底：不许产出单字查询（"那吗"这类；单字量词上面那组已覆盖，这是防漏网）
            # 【已知边界·登记而不修（批次 26-C 裁决）】这条兜底只判"剥完剩 1 个字符"，
            # 因此病态输入会留下脏尾巴：
            #   "那那" 剥掉一个"那"剩"那"（长度 1）→ 拦下 → 主题 那那
            #   "那吗" 剥掉"那"剩"吗"（长度 1）→ 拦下 → 主题 那吗
            # 两者都不如"直接返回主题词"干净。之所以不为它们再加一条分支
            # （例如"剩余全部由指代词构成 ⇒ 整问都是指代"）：真实追问不会长成这样，
            # 为一个不可能自然出现的输入增加判定路径，收益不抵复杂度与回归面。
            # 行为已被 backend/tests/test_query_rewrite.py 的参数化用例逐条钉住，
            # 将来若要改，必须显式改动那些用例，不会"顺手"漂移。
            if len(payload) < 2:
                break
            # 这一刀会把词切成单字碎片就放弃（"那个员工" -"那"→ "个员工"）。
            # 例外："那这个呢？" 剥完得到的 "这个呢？" 虽然也以"这"开头，但它本身
            # 就是另一个前缀，能整词剥干净，不该拦 —— 拦了会退回 "主题 那这个"。
            # 注意用元组 _REFERENCE_PREFIXES：str.startswith 不接受 list。
            if payload[:1] in _FRAGMENT_LEAD_CHARS and not candidate.startswith(
                _REFERENCE_PREFIXES
            ):
                break
            # 每次至少吃掉 1 个字符，循环必然终止
            remainder = candidate
        # 依次清理：开头标点、结尾标点、语气词"呢"
        remainder = _strip_question_noise(remainder)
        # 剥完指代后什么都不剩（如"那这个呢？"）→ 整问都是主题
        if not remainder:
            return topic
        # 主题在前、追问剩余在后。
        # 批次 24 A/B（10 条 followup_rewrite + 5 条 multi_turn，加/不加尾巴各跑三轮）：
        # 追加"怎么算"会让 multiturn-003 / 004 的第 2 轮排名各掉 1 位
        # （4→3、3→2），15 题子集 MRR@10 从 0.6444 降到 0.6278，其余 13 题逐题不变、
        # 三轮结果逐位复现 —— 无收益且有损，故不再追加。
        return f"{topic} {remainder}"
