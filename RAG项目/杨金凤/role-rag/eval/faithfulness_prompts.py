"""Faithfulness 自定义 prompt：忽略引用格式词/礼貌用语，只评实质内容是否被上下文支持。

RAGAS 0.3.9 的 Faithfulness 由两个 PydanticPrompt 组成：
statement_generator_prompt（拆陈述句）+ nli_statements_prompt（逐句判 NLI）。
这里子类化原 prompt，只覆盖 instruction，复用原 input/output model 与 examples。
"""
from __future__ import annotations

from ragas.metrics import Faithfulness
from ragas.metrics._faithfulness import NLIStatementPrompt, StatementGeneratorPrompt


class _LenientStatementGeneratorPrompt(StatementGeneratorPrompt):
    instruction = (
        "给定问题和答案，把答案拆解为一条条独立的实质陈述句，不使用代词。"
        "不要单独输出「根据资料N」「第X页」这类纯引用标记作为陈述句——"
        "把它们合并进对应实质内容或直接跳过。以 JSON 格式输出。"
    )


class _LenientNLIStatementPrompt(NLIStatementPrompt):
    instruction = (
        "你的任务是判断一系列陈述句是否忠实于给定上下文。对每个陈述句返回 verdict："
        "1 表示该陈述句的实质内容可以直接从上下文推断，0 表示不能。"
        "注意：忽略「根据资料N」「第X页」「结论：」这类引用格式标记，以及「您好」等礼貌用语——"
        "只评估实质的医学内容是否能在上下文中找到。"
    )


def build_faithfulness() -> Faithfulness:
    """构造用了宽松 prompt 的 faithfulness 指标（不发起任何 LLM 调用）。"""
    return Faithfulness(
        nli_statements_prompt=_LenientNLIStatementPrompt(),
        statement_generator_prompt=_LenientStatementGeneratorPrompt(),
    )
