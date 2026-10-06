# 契约的作用是"模型乱写时我们能明确地失败"，所以测试的重点在拒绝而不是接受。
import pytest
from pydantic import ValidationError

from app.generation.schema import Answer, Citation, answer_parser


def test_parses_well_formed_answer():
    raw = ('{"status": "ok", "answer": "可以要求支付逾期利息。",'
           ' "citations": [{"law": "中华人民共和国民法典", "article": "第五百七十九条",'
           ' "paragraph": null, "item": null, "quote": "当事人一方未支付价款、报酬、租金、利息"}],'
           ' "disclaimer": ""}')
    parsed = answer_parser().parse(raw)
    assert parsed.status == "ok"
    assert parsed.citations[0].article == "第五百七十九条"


def test_rejects_unknown_status():
    # status 只有三个合法值；把 error/abstain 留给编排层是刻意的，
    # 模型若能自称"未找到依据"，就等于给了它一条绕过校验的路
    with pytest.raises(ValidationError):
        Answer.model_validate({"status": "error", "answer": "x", "citations": []})


def test_rejects_abstain_status_from_model():
    with pytest.raises(ValidationError):
        Answer.model_validate({"status": "abstain", "answer": "x", "citations": []})


def test_citation_requires_quote():
    with pytest.raises(ValidationError):
        Citation(law="民法典", article="第五百八十四条")


def test_answer_defaults_are_empty_not_missing():
    # 模型可能只给 answer 不给 disclaimer，缺字段不该让整条答案报废
    parsed = Answer.model_validate({"status": "ok", "answer": "正文", "citations": []})
    assert parsed.disclaimer == ""
    assert parsed.citations == []


def test_parser_rejects_non_json_text():
    from langchain_core.exceptions import OutputParserException
    with pytest.raises(OutputParserException):
        answer_parser().parse("对不起，我不能回答这个问题。")


def test_parser_format_instructions_mention_json():
    # DeepSeek 的 json_object 模式要求提示词里出现 json 这个词，
    # 格式说明进 system 提示，正好满足（见设计文档 4.5）
    assert "json" in answer_parser().get_format_instructions().lower()
