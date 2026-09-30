# -*- coding: utf-8 -*-
"""检索页码匹配口径测试

核心规则：chunk 覆盖的**任一页**命中金标准即算命中。
因为答案正文可能位于跨页块的后半部分（被记为首页），只看首页会漏判。
"""

import pytest

from scripts.ragas_eval import _pages_match, compute_retrieval_metrics


def test_单页块行为不变():
    assert _pages_match(("42581", (6,)), ("42581", 6)) is True
    assert _pages_match(("42581", (6,)), ("42581", 7)) is False


def test_跨页块的次页命中被正确计入():
    """这是本次修正的核心：page_no=6 但 page_nums=[6,7] 的块，应命中金标准第 7 页"""
    assert _pages_match(("32904", (6, 7)), ("32904", 7)) is True
    assert _pages_match(("32904", (6, 7)), ("32904", 6)) is True


def test_跨页块未覆盖的页仍不命中():
    assert _pages_match(("32904", (6, 7)), ("32904", 8)) is False


def test_文件名包含匹配保持生效():
    long_name = "GBT 42581-2023 信息技术服务 数据中心业务连续性等级评价准则.pdf"
    assert _pages_match((long_name, (6,)), ("42581", 6)) is True
    assert _pages_match((long_name, (6,)), ("41479", 6)) is False


def test_空页码列表不命中():
    assert _pages_match(("32904", ()), ("32904", 6)) is False


def test_指标计算_跨页块在首位即_rr_为1():
    gold = [("32904", 7)]
    retrieved = [("32904", (6, 7)), ("32904", (20,))]
    m = compute_retrieval_metrics(retrieved, gold, [1, 3, 5])
    assert m["recall@1"] == 1.0
    assert m["rr"] == pytest.approx(1.0)


def test_指标计算_跨页块在第三位_recall1为0_recall3为1():
    gold = [("32904", 7)]
    retrieved = [("32904", (20,)), ("32904", (10,)), ("32904", (6, 7))]
    m = compute_retrieval_metrics(retrieved, gold, [1, 3, 5])
    assert m["recall@1"] == 0.0
    assert m["recall@3"] == 1.0
    assert m["rr"] == pytest.approx(1 / 3)
