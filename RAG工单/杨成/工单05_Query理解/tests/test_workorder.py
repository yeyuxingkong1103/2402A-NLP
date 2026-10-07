from common.query import analyze_query
from main import build_query_context


def test_query_extracts_company_and_year():
    result = analyze_query("2024 年平安银行营业收入是多少？")
    assert result.intent == "factual"
    assert "2024" in result.entities
    assert "平安银行" in result.entities


def test_context_uses_recent_turn():
    context = build_query_context("它是多少？", [("旧问题", "旧答案"), ("营业收入？", "100 亿元")], limit=1)
    assert "100 亿元" in context and "旧答案" not in context
