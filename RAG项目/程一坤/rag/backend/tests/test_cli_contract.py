"""命令行入口的字段契约回归测试。

为什么需要它：demo_ask 不在任何模块测试的覆盖范围内，
上一轮 chat/service.py 改了法源字段名后，命令行直接 KeyError 崩溃，
而全量测试依然全绿。这里用真实字段结构锁住契约，防止再犯。
"""

import pytest

from app.chat.result import ChatResult
from app.cli.demo_ask import print_sources


def test_print_sources_accepts_real_chat_service_fields(capsys) -> None:
    """法源字段必须是 ChatService 实际产出的那套（law_name / article_number / paragraph_number）。"""
    sources = [
        {
            "chunk_id": "chunk-1",
            "law_name": "中华人民共和国劳动合同法",
            "article_number": "第四十七条",
            "paragraph_number": "1",
            "page": None,
        },
        {
            "chunk_id": "chunk-2",
            "law_name": "中华人民共和国劳动法",
            "article_number": None,
            "paragraph_number": None,
            "page": None,
        },
    ]

    print_sources(sources)
    output = capsys.readouterr().out

    assert "【引用法源】共 2 条" in output
    # 完整写法"第四十七条"不应再被包一层"第…条"
    assert "《中华人民共和国劳动合同法》第四十七条第 1 款" in output
    # 条号缺失时如实标注，不用别的字段顶替
    assert "未识别条号" in output
    assert "None" not in output


def test_format_article_label_supports_both_forms() -> None:
    """条号两种形态（完整写法 / 纯编号）都要正确显示，且带款号。"""
    from app.cli.demo_ask import format_article_label

    assert format_article_label("第四十七条") == "第四十七条"
    assert format_article_label("47") == "第 47 条"
    assert format_article_label("第四十七条", "2") == "第四十七条第 2 款"
    assert format_article_label(None) == "未识别条号"


def test_print_sources_handles_missing_law_name(capsys) -> None:
    """法规名缺失时不能崩，也不能打印出 None。"""
    print_sources([{"chunk_id": "chunk-x", "law_name": None, "article_number": None}])
    output = capsys.readouterr().out

    assert "未知法规" in output
    assert "None" not in output


def test_chat_result_exposes_fields_used_by_cli() -> None:
    """CLI 读取的字段必须在 ChatResult 上真实存在。"""
    for field in ("answer", "sources", "refused", "retrieval_stats", "guardrail_applied"):
        assert hasattr(ChatResult, "__dataclass_fields__"), "ChatResult 应为 dataclass"
        assert field in ChatResult.__dataclass_fields__, f"ChatResult 缺少 CLI 依赖的字段：{field}"


def test_print_sources_accepts_empty_list(capsys) -> None:
    """空法源列表时静默返回（拒答场景会走到这里）。"""
    print_sources([])
    assert capsys.readouterr().out == ""
