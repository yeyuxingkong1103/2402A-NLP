"""测试法规元数据处理工具。"""

import pytest

from app.ingest.article_number_rules import build_article_path


def test_build_article_path_single_article() -> None:
    """测试只有条号的情况。"""
    assert build_article_path("47") == "47"


def test_build_article_path_with_paragraph() -> None:
    """测试条 + 款的情况。"""
    assert build_article_path("47", "2") == "47-2"


def test_build_article_path_with_paragraph_and_item() -> None:
    """测试条 + 款 + 项的情况。"""
    assert build_article_path("47", "2", "1") == "47-2-1"


def test_build_article_path_empty_article_number() -> None:
    """测试条号为空串应抛出 ValueError。"""
    with pytest.raises(ValueError, match="条号不能为空"):
        build_article_path("")


def test_build_article_path_chinese_article_number() -> None:
    """测试条号为中文应抛出 ValueError。"""
    with pytest.raises(ValueError, match="条号必须是阿拉伯数字编号.*入库前必须先规范化"):
        build_article_path("第四十七条")


def test_build_article_path_with_zhi() -> None:
    """测试「之一」条款。"""
    assert build_article_path("99之1") == "99之1"
    assert build_article_path("99之1", "1") == "99之1-1"


def test_build_article_path_item_without_paragraph() -> None:
    """测试有项号但无款号应抛出 ValueError。"""
    with pytest.raises(ValueError, match="有项号时必须有款号"):
        build_article_path("47", None, "1")


def test_build_article_path_empty_paragraph() -> None:
    """测试款号为空串应抛出 ValueError。"""
    with pytest.raises(ValueError, match="款号不能为空或只有空白"):
        build_article_path("47", "")


def test_build_article_path_whitespace_paragraph() -> None:
    """测试款号为纯空白应抛出 ValueError。"""
    with pytest.raises(ValueError, match="款号不能为空或只有空白"):
        build_article_path("47", "   ")


def test_build_article_path_chinese_paragraph() -> None:
    """测试款号为中文应抛出 ValueError。"""
    with pytest.raises(ValueError, match="款号必须是阿拉伯数字编号.*入库前必须先规范化"):
        build_article_path("47", "二")


def test_build_article_path_empty_item() -> None:
    """测试项号为空串应抛出 ValueError。"""
    with pytest.raises(ValueError, match="项号不能为空或只有空白"):
        build_article_path("47", "2", "")


def test_build_article_path_whitespace_item() -> None:
    """测试项号为纯空白应抛出 ValueError。"""
    with pytest.raises(ValueError, match="项号不能为空或只有空白"):
        build_article_path("47", "2", "   ")


def test_build_article_path_chinese_item() -> None:
    """测试项号为中文应抛出 ValueError。"""
    with pytest.raises(ValueError, match="项号必须是阿拉伯数字编号.*入库前必须先规范化"):
        build_article_path("47", "2", "一")


def test_build_article_path_double_hyphen() -> None:
    """测试空款号+项号会产生双连字符应抛出 ValueError。"""
    with pytest.raises(ValueError, match="款号不能为空或只有空白"):
        build_article_path("47", "", "1")


def test_build_article_path_negative_paragraph() -> None:
    """测试负号款号应抛出 ValueError。"""
    with pytest.raises(ValueError, match="款号必须是阿拉伯数字编号.*入库前必须先规范化"):
        build_article_path("47", "-1")


def test_build_article_path_non_numeric_item() -> None:
    """测试非数字项号应抛出 ValueError。"""
    with pytest.raises(ValueError, match="项号必须是阿拉伯数字编号.*入库前必须先规范化"):
        build_article_path("47", "2", "abc")


def test_build_article_path_leading_zeros() -> None:
    """测试前导零会被去除。"""
    assert build_article_path("0047") == "47"
    assert build_article_path("0047", "002") == "47-2"
    assert build_article_path("0047", "002", "001") == "47-2-1"
    assert build_article_path("99之01") == "99之1"
    assert build_article_path("99之01", "002") == "99之1-2"


def test_build_article_path_all_zeros() -> None:
    """测试全是 0 的情况。"""
    assert build_article_path("0") == "0"
    assert build_article_path("00") == "0"
    assert build_article_path("000", "00", "0") == "0-0-0"


def test_build_article_path_zhi_missing_after() -> None:
    """测试「之」后缺失数字应抛出 ValueError。"""
    with pytest.raises(ValueError, match="「之」前后必须都有数字"):
        build_article_path("99之")


def test_build_article_path_zhi_missing_before() -> None:
    """测试「之」前缺失数字应抛出 ValueError。"""
    with pytest.raises(ValueError, match="「之」前后必须都有数字"):
        build_article_path("之1")


def test_build_article_path_multiple_zhi() -> None:
    """测试多个「之」应抛出 ValueError。"""
    with pytest.raises(ValueError, match="最多只能包含一个「之」字"):
        build_article_path("1之2之3")


def test_build_article_path_fullwidth_digit() -> None:
    """测试全角数字应抛出 ValueError。"""
    with pytest.raises(ValueError, match="条号必须是阿拉伯数字编号.*入库前必须先规范化"):
        build_article_path("４７")


def test_build_article_path_fullwidth_leading_zeros() -> None:
    """测试全角前导零应抛出 ValueError。"""
    with pytest.raises(ValueError, match="条号必须是阿拉伯数字编号.*入库前必须先规范化"):
        build_article_path("００７")


def test_build_article_path_all_zeros_preserved() -> None:
    """测试全零特例：'00' → '0'（明确设计）。"""
    assert build_article_path("00") == "0"


def test_build_article_path_single_number_too_long() -> None:
    """测试单个编号超过 16 字符应抛出 ValueError。"""
    with pytest.raises(ValueError, match="条号长度不能超过 16 字符"):
        build_article_path("9" * 17)


def test_build_article_path_final_path_too_long() -> None:
    """测试最终 article_path 超过 128 字符应抛出 ValueError。"""
    with pytest.raises(ValueError, match="条号长度不能超过 16 字符"):
        build_article_path("9" * 200)
