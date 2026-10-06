# -*- coding: utf-8 -*-
"""输入校验测试

QueryRequest 已在长度上做了约束（1-500）；本任务补齐「纯空白」「纯符号」等
语义为空的输入 —— 它们能通过长度校验，但对检索毫无意义，且会让模型产出
无意义答案。
"""

import pytest
from pydantic import ValidationError

from backend.api.query import QueryRequest


def test_正常问题通过校验():
    r = QueryRequest(question="数据中心业务连续性分为几个等级？")
    assert r.question.startswith("数据中心")


def test_纯空白被拒绝():
    with pytest.raises(ValidationError):
        QueryRequest(question="   ")


def test_首尾空白被自动去除():
    r = QueryRequest(question="  数据中心业务连续性分为几个等级？  ")
    assert r.question == "数据中心业务连续性分为几个等级？"


def test_纯标点被拒绝():
    with pytest.raises(ValidationError):
        QueryRequest(question="？？？！！！")


def test_超长问题被拒绝():
    with pytest.raises(ValidationError):
        QueryRequest(question="问" * 501)


def test_上限长度恰好通过():
    r = QueryRequest(question="问" * 500)
    assert len(r.question) == 500


def test_正常问题中的标点不受影响():
    r = QueryRequest(question="GB/T 32904 中，功能充分性属于哪个质量特性的子特性？")
    assert "GB/T" in r.question


# ---------------------------------------------------------------------------
# 长度边界：校验器为 mode="before"，长度约束作用于 strip **之后**的文本，
# 因此首尾空白不计入 500 字上限（符合「可见字数 ≤ 500」的直觉）。
# ---------------------------------------------------------------------------

def test_首尾空白不计入长度上限():
    # 可见 500 字 + 首尾各两个空格：strip 后恰好 500 字，应当通过
    r = QueryRequest(question="  " + "问" * 500 + "  ")
    assert len(r.question) == 500
    assert r.question == "问" * 500


def test_可见498字带首尾空白也通过():
    r = QueryRequest(question="  " + "问" * 498 + "  ")
    assert len(r.question) == 498


def test_strip_后仍超长则被拒绝():
    with pytest.raises(ValidationError):
        QueryRequest(question="  " + "问" * 501 + "  ")


def test_非字符串输入被拒绝():
    """类型防护：mode="before" 下 value 可能不是字符串，应交给类型校验报错"""
    with pytest.raises(ValidationError):
        QueryRequest(question=123)
