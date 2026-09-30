"""测试查询改写与检索链路接入。"""

import logging

import pytest

from app.retrieval.query_rewrite import QueryRewriteResult, QueryRewriter
from app.retrieval.service import RetrievalService


class CapturingVectorRetriever:
    """记录检索问题的替身。"""

    def __init__(self) -> None:
        self.questions: list[str] = []

    def retrieve(self, question: str, **kwargs):
        self.questions.append(question)
        return []


class FakeMemory:
    """提供短期消息读取的替身。"""

    def __init__(self, messages):
        self.messages = messages

    def read_messages(self, user_id: str, session_id: str):
        return self.messages


def test_rewrites_implicit_reference_using_latest_user_topic() -> None:
    result = QueryRewriter().rewrite(
        "那 12 期呢",
        [
            {"role": "user", "content": "经济补偿怎么算"},
            {"role": "assistant", "content": "要看工作年限。"},
        ],
    )

    assert result.changed is True
    # 批次 24：改写结果不再追加"怎么算"尾巴（A/B 三轮复现：加了会让 15 题子集
    # MRR@10 从 0.6444 降到 0.6278，故去掉；详见 query_rewrite._compose_query）
    assert result.rewritten_query == "经济补偿 12 期"
    assert "那" in result.reasons
    assert "主题词：经济补偿" in result.reasons


def test_returns_original_question_when_no_reference_is_detected() -> None:
    question = "解除劳动合同需要提前多久通知？"

    result = QueryRewriter().rewrite(
        question,
        [{"role": "user", "content": "劳动合同解除条件"}],
    )

    assert result == QueryRewriteResult(
        original_query=question,
        rewritten_query=question,
        changed=False,
        reasons=[],
    )


def test_skips_malformed_empty_and_system_messages() -> None:
    question = "那 12 期呢"
    result = QueryRewriter().rewrite(
        question,
        [
            {"role": "system", "content": "系统规则"},
            {"role": "user"},
            "not-a-message",
            {"role": "assistant", "content": ""},
            {"role": "user", "content": "经济补偿怎么算"},
        ],
    )

    assert result.changed is True
    assert "经济补偿" in result.rewritten_query


def test_returns_original_for_empty_one_message_or_system_only_context() -> None:
    rewriter = QueryRewriter()
    question = "那 12 期呢"

    assert rewriter.rewrite(question, []).rewritten_query == question
    assert rewriter.rewrite(question, [{"role": "user", "content": "主题"}]).rewritten_query == question
    assert rewriter.rewrite(question, [{"role": "system", "content": "规则"}]).rewritten_query == question


def test_rewrite_failure_falls_back_to_original_question() -> None:
    question = "那 12 期呢"

    result = QueryRewriter().rewrite(question, [{"role": "user", "content": "!!!"}])

    assert result.rewritten_query == question
    assert result.changed is False


class FailingMemory:
    def read_messages(self, user_id: str, session_id: str):
        raise RuntimeError("redis://user:secret@host/9")


def test_rewrite_failure_logs_type_and_request_id_without_secret(caplog) -> None:
    service = RetrievalService(
        vector_retriever=CapturingVectorRetriever(),
        query_rewriter=QueryRewriter(),
        short_term_memory=FailingMemory(),
    )

    with caplog.at_level(logging.WARNING):
        result = service.retrieve(
            "那 12 期呢",
            user_id="user-a",
            session_id="session-1",
            request_id="req_rewrite123",
        )

    assert result.query_rewrite.changed is False
    assert "secret" not in caplog.text
    record = next(record for record in caplog.records if "查询改写失败" in record.message)
    assert record.request_id == "req_rewrite123"


def test_retrieval_uses_rewritten_query_when_short_term_context_is_available() -> None:
    retriever = CapturingVectorRetriever()
    service = RetrievalService(
        vector_retriever=retriever,
        query_rewriter=QueryRewriter(),
        short_term_memory=FakeMemory(
            [
                {"role": "user", "content": "经济补偿怎么算"},
                {"role": "assistant", "content": "按工作年限确定。"},
            ]
        ),
    )

    service.retrieve("那 12 期呢", user_id="user-a", session_id="session-1")

    # 批次 24：同上，改写结果不再带"怎么算"尾巴
    assert retriever.questions == ["经济补偿 12 期"]


def test_retrieval_keeps_original_query_without_rewrite_signal() -> None:
    retriever = CapturingVectorRetriever()
    service = RetrievalService(
        vector_retriever=retriever,
        query_rewriter=QueryRewriter(),
        short_term_memory=FakeMemory(
            [
                {"role": "user", "content": "经济补偿怎么算"},
                {"role": "assistant", "content": "按工作年限确定。"},
            ]
        ),
    )

    service.retrieve(
        "解除劳动合同需要提前多久通知？",
        user_id="user-a",
        session_id="session-1",
    )

    assert retriever.questions == ["解除劳动合同需要提前多久通知？"]


# 单字碎片：这些字不能单独成词，落在改写结果「主题」之后的开头就是脏查询
# （批次 24b ① 引入 个/的；批次 26-C 按裁决扩到"量词/助词单字"）。
# 这里刻意独立写一份（不复用生产常量），让测试自己重申一遍不变量。
_FRAGMENT_CHARS = {"个", "的", "些", "种", "位", "名", "只", "条"}
_FRAGMENT_MESSAGES = [
    {"role": "user", "content": "经济补偿怎么算"},
    {"role": "assistant", "content": "按工作年限确定。"},
]


def _remainder_after_topic(rewritten_query: str) -> str:
    """取改写结果里「主题」之后的部分；整问都是主题（没拼剩余）时返回空串。"""
    parts = rewritten_query.split(" ", 1)
    return parts[1] if len(parts) == 2 else ""


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        # ⑤ 的落地目标不能被新约束破坏：剥干净后整问都是主题
        ("那这个呢？", "经济补偿"),
        # 循环剥离会切出单字碎片 —— 按新约束「宁可不剥」，保留完整词
        ("那那那个怎么算", "经济补偿 那个怎么算"),
        ("那个员工的工资怎么算？", "经济补偿 那个员工的工资怎么算"),
        ("前面那个员工的工资怎么算？", "经济补偿 那个员工的工资怎么算"),
        # 批次 26-C 扩充：单个量词留在查询开头同样是碎片
        # （"那些情况怎么算" 改前产出「经济补偿 些情况怎么算」，以"些"打头）
        ("那些情况怎么算", "经济补偿 那些情况怎么算"),
        ("那种怎么算", "经济补偿 那种怎么算"),
        ("那位员工呢", "经济补偿 那位员工"),
        ("那名员工呢", "经济补偿 那名员工"),
        ("那只呢", "经济补偿 那只"),
        ("那条呢", "经济补偿 那条"),
        # 兜底规则「剩余长度 < 2 不剥」：剥完只剩单字就不再剥
        ("那吗", "经济补偿 那吗"),
        ("那那", "经济补偿 那那"),
        # 剥完什么都不剩 ⇒ 整问都是主题（"不剥"不适用于这种情况）
        ("那", "经济补偿"),
        ("那这个呢？", "经济补偿"),
        # 剩余开头不是碎片，照旧剥干净（确认没误伤正常路径）
        ("那这个期间怎么算？", "经济补偿 期间怎么算"),
        ("那 12 期呢", "经济补偿 12 期"),
        ("上述情况如何处理？", "经济补偿 情况如何处理"),
    ],
)
def test_compose_query_does_not_split_words_into_fragment(question: str, expected: str) -> None:
    """批次 24b ① + 批次 26-C：剥指代前缀不得把词切成单字碎片。

    约束是「宁可不剥」——`前面那个员工的工资` 剥掉「前面」会剩 `个员工的工资`，
    这种脏查询拿去检索不如原样保留，遇到就放弃这一刀。
    批次 26-C 把碎片首字扩到量词/助词单字组，并加了「剩余长度 < 2 不剥」的兜底；
    注意兜底**不覆盖**"剥完什么都不剩"的情形——那是"整问都是指代"，应当返回主题词。
    """
    result = QueryRewriter().rewrite(question, _FRAGMENT_MESSAGES)

    assert result.changed is True
    assert result.rewritten_query == expected


@pytest.mark.parametrize(
    "question",
    [
        "那那那个怎么算",
        "那个员工的工资怎么算？",
        "前面那个员工的工资怎么算？",
        "那 12 期呢",
        "这个怎么算",
        "那这个期间怎么算？",
        "那些情况怎么算",
        "那种怎么算",
        "那位员工呢",
        "那名员工呢",
        "那只呢",
        "那条呢",
        "前面前面那个怎么算",
        "前面那些条款怎么算",
    ],
)
def test_rewrite_never_starts_remainder_with_fragment(question: str) -> None:
    """锁住不变量：改写结果「主题」之后的部分，不得以单字碎片开头。"""
    rewritten = QueryRewriter().rewrite(question, _FRAGMENT_MESSAGES).rewritten_query

    assert _remainder_after_topic(rewritten)[:1] not in _FRAGMENT_CHARS


def test_fragment_guard_keeps_the_whole_word_when_strip_would_fragment_it() -> None:
    """护栏的行为是"放弃这一刀"，不是"把碎片删掉"——原词必须完整保留。"""
    rewritten = QueryRewriter().rewrite("那些情况怎么算", _FRAGMENT_MESSAGES).rewritten_query

    # 「那些」完整保留（不是"些"、也不是"些情况"）
    assert "那些情况" in rewritten
    assert rewritten == "经济补偿 那些情况怎么算"


@pytest.mark.parametrize(
    ("question", "expected", "why"),
    [
        ("那那", "经济补偿 那那", "剥掉一个「那」只剩单字「那」，按「剩余长度 < 2 不剥」拦下"),
        ("那吗", "经济补偿 那吗", "剥掉「那」只剩语气词「吗」，同样属于单字剩余"),
    ],
)
def test_known_boundary_pathological_input_keeps_dirty_tail(
    question: str, expected: str, why: str
) -> None:
    """【已知边界，登记而不修】病态输入的脏尾巴是有意保留的行为。

    这两条输入不是真实追问，而是"指代词 + 单字"的退化形态：兜底规则只看
    「剥完是否只剩 1 个字符」，于是它们退回"主题 + 原样剩余"——结果比
    "直接返回主题词"脏，但也不构成错误查询（检索侧仍能靠主题词命中）。

    批次 26-C 裁决：不为病态输入增加判定分支（收益不抵复杂度与回归面），
    登记为已知边界即可。本用例的作用就是**把边界钉住**：
    将来若有人想让它变干净，改动必须显式改到这里，不会被顺手带过。
    """
    result = QueryRewriter().rewrite(question, _FRAGMENT_MESSAGES)

    assert result.changed is True, f"应走改写路径（{why}）"
    assert result.rewritten_query == expected
    # 脏尾巴的成因是"单字剩余"：剩余部分 = 指代词 + 1 个字符，
    # 剥掉那个指代词后只剩 1 字，正是「剩余长度 < 2 不剥」拦下的原因。
    remainder = _remainder_after_topic(result.rewritten_query)
    assert remainder.startswith("那")
    assert len(remainder[len("那"):]) == 1, f"剩余部分应为「指代词 + 单字」，实际 {remainder!r}"


