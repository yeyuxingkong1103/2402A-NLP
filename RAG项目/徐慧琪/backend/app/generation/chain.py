"""LCEL 生成链：prompt | llm | parser。

存在的理由（技术方案 2.2）：生成段用框架的编排能力，业务规则不进来。
链只有这三节；校验、重生成、降级都在 answer.py 的 Python 编排里——
把它们塞进 LCEL 的 with_fallbacks 会把"带原因重试一次再降级"表达得极难读。
"""
from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.generation.profiles import system_prompt
from app.generation.schema import answer_parser


def _as_literal(text: str) -> str:
    """把要原样出现的文本转义成模板字面量。

    格式说明是**含 JSON schema 的文本**（成片花括号），照直拼进
    ChatPromptTemplate 会被当模板解析：`{"status": ...}` 的冒号被读成格式
    说明符，实测抛 `ValueError: Invalid format specifier in f-string template`。
    转义后渲染结果与原文逐字一致（test_answer 有一条断言盯着），
    故提示词本身一个字都不用改——这是修法里最小的一种。
    """
    return text.replace("{", "{{").replace("}", "}}")


def build_prompt(side: str) -> ChatPromptTemplate:
    """system 里带上格式说明：Ollama 的 json 模式只管"是 JSON"，
    管不了字段名，字段契约仍要写进提示词。

    转义盖住整条 system 而不只是格式说明：两侧提示词现在还干净，但提示词是
    最常改的文件，将来往举例里写一个 `{"status": "ok"}` 就会重蹈同一个崩溃。
    human 里的 {payload} 是**变量**、要留着让调用方填，不能一起转义。
    """
    instructions = answer_parser().get_format_instructions()
    return ChatPromptTemplate.from_messages([
        ("system", _as_literal(system_prompt(side) + "\n\n" + instructions)),
        ("human", "{payload}"),
    ])


def build_chain(llm, side: str):
    """返回可调用的链。

    llm 是任务参数而非模块单例：测试要能塞假模型，生产由 llm_router 提供。
    解析器接在链尾（技术方案 2.2：解析失败即失败，不兜底编造）；
    真正的校验关是独立的 cite_verify，串在这里的是它的调用方职责。
    """
    return build_prompt(side) | llm | StrOutputParser()
