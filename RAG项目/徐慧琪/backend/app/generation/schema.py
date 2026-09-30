"""生成层的输出契约。

存在的理由：解析失败必须能被**明确地**判为失败并触发重生成，而不是
"尽力猜模型想说什么"。猜出来的答案绕过了校验关，等于把安全阀拆了。

status 的三个值（ok / need_more_info / out_of_scope）是模型可选的；
error 与 abstain 由编排层单方面赋给调用方，**模型无权输出**——
否则模型可以自称"未找到依据"来规避引用校验。
"""
from __future__ import annotations

from typing import Literal

from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


class Citation(BaseModel):
    """一条引用。article 允许模型按自己的写法给（中文数字或阿拉伯），
    校验时统一走 query_parse 归一化，不在这里限制写法。"""
    law: str = ""
    article: str = Field(..., description="条号，如「第五百八十四条」或「584」")
    paragraph: int | None = Field(None, description="款号，整条引用时留空")
    item: str | None = Field(None, description="项号，如「一」；无项时留空")
    quote: str = Field(..., description="必须逐字来自所给法条原文的片段")


class Answer(BaseModel):
    """模型输出契约。默认值取空而非必填——缺字段不该让整条答案报废，
    校验关自然会因为"没有引用"而判它失败。"""
    status: Literal["ok", "need_more_info", "out_of_scope"] = "ok"
    answer: str = ""
    citations: list[Citation] = []
    disclaimer: str = ""


def answer_parser() -> PydanticOutputParser:
    """契约的解析器工厂。每次新建：parser 带可变状态，共享会串味。"""
    return PydanticOutputParser(pydantic_object=Answer)
