"""测试时效过滤功能"""
from datetime import date, datetime, timezone
import pytest
from app.retrieval.filters import (
    build_filter_expression,
    to_timestamp,
    FilterError,
    UNKNOWN_DATE_TIMESTAMP,
)


def test_to_timestamp_with_date():
    """测试日期转时间戳"""
    d = date(2023, 6, 15)
    ts = to_timestamp(d)
    # 验证是 UTC 零点的时间戳
    expected = int(datetime(2023, 6, 15, tzinfo=timezone.utc).timestamp())
    assert ts == expected


def test_to_timestamp_with_string():
    """测试字符串日期转时间戳"""
    ts = to_timestamp("2023-06-15")
    expected = int(datetime(2023, 6, 15, tzinfo=timezone.utc).timestamp())
    assert ts == expected


def test_to_timestamp_invalid_format():
    """测试非法日期格式"""
    with pytest.raises(FilterError, match="日期格式应为 YYYY-MM-DD"):
        to_timestamp("2023/06/15")


def test_to_timestamp_none_returns_none():
    """测试 None 原样返回"""
    assert to_timestamp(None) is None


def test_build_filter_no_conditions():
    """测试没有任何条件时返回 None"""
    expr = build_filter_expression()
    assert expr is None


def test_build_filter_jurisdiction_only():
    """测试只有法域过滤"""
    expr = build_filter_expression(jurisdiction="中国大陆")
    assert expr == 'jurisdiction == "中国大陆"'


def test_build_filter_jurisdiction_empty_raises_error():
    """测试法域为空字符串时报错"""
    with pytest.raises(FilterError, match="法域不能是空字符串"):
        build_filter_expression(jurisdiction="")


def test_build_filter_document_types():
    """测试文书类型过滤"""
    expr = build_filter_expression(document_types=["法律", "行政法规"])
    assert 'document_type in ["法律", "行政法规"]' in expr


def test_build_filter_only_current():
    """测试只要现行有效"""
    expr = build_filter_expression(only_current=True)
    assert expr == "is_current == true"


def test_build_filter_escapes_string_literals():
    expr = build_filter_expression(
        jurisdiction='中国"大陆\\测试',
        document_types=['法"律', "行政\\法规"],
    )

    assert 'jurisdiction == "中国\\"大陆\\\\测试"' in expr
    assert 'document_type in ["法\\"律", "行政\\\\法规"]' in expr


def test_build_filter_as_of_date_includes_unknown():
    """测试时间点过滤，包含未知生效日期"""
    ts = to_timestamp("2023-06-15")
    expr = build_filter_expression(as_of_date="2023-06-15", include_unknown_effective_date=True)

    # 应该包含：生效日期未知(0) 或 生效日期<=时间点
    assert f"(effective_date == {UNKNOWN_DATE_TIMESTAMP} or effective_date <= {ts})" in expr
    # 应该包含：失效日期为空 或 失效日期>时间点
    assert f"(expiration_date is null or expiration_date > {ts})" in expr


def test_build_filter_as_of_date_excludes_unknown():
    """测试时间点过滤，排除未知生效日期"""
    ts = to_timestamp("2023-06-15")
    expr = build_filter_expression(as_of_date="2023-06-15", include_unknown_effective_date=False)

    # 应该排除生效日期未知(0)
    assert f"(effective_date != {UNKNOWN_DATE_TIMESTAMP} and effective_date <= {ts})" in expr


def test_build_filter_multiple_conditions():
    """测试组合多个条件"""
    expr = build_filter_expression(
        as_of_date="2023-06-15",
        jurisdiction="中国大陆",
        document_types=["法律"],
        only_current=True,
    )

    # 所有条件都应该出现
    assert 'jurisdiction == "中国大陆"' in expr
    assert 'document_type in ["法律"]' in expr
    assert "is_current == true" in expr
    assert "effective_date" in expr
    assert "expiration_date" in expr

    # 用 and 连接
    assert " and " in expr
