# 工单编号：人工智能NLP-RAG-Query理解优化
#          （工单 PDF 原文：人工智能NLP-RAG-Query 理解优化任务）
"""组织结构图：文字合并、绑定、建树、聚合。"""
import os

import pytest

from src import config, orgchart_parser


PDF = os.path.join(config.ROOT, "招股说明书2.pdf")
pytestmark = pytest.mark.slow          # 需要读 PDF


# --- 文字合并（竖排是这道题的关键）-----------------------------------------
def test_merge_vertical_text():
    """「财务部」在 PDF 里是三个逐字堆叠的 word，必须合并成一个标签。"""
    words = [(142.0, 349.3, 152.5, 359.8, "财"),
             (142.0, 359.8, 152.5, 370.3, "务"),
             (142.0, 370.3, 152.5, 380.8, "部")]
    labels = orgchart_parser.merge_words_to_labels(words, gap=5.0)
    assert [t for t, _ in labels] == ["财务部"]


def test_merge_multiline_horizontal():
    """「薪酬与考核委员会」是两行横排，必须按行合并。"""
    words = [(144.6, 88.5, 197.1, 99.0, "薪酬与考核"),
             (155.1, 102.1, 186.6, 112.6, "委员会")]
    labels = orgchart_parser.merge_words_to_labels(words, gap=5.0)
    assert [t for t, _ in labels] == ["薪酬与考核委员会"]


def test_merge_keeps_distant_words_apart():
    words = [(100.0, 100.0, 150.0, 112.0, "股东大会"),
             (400.0, 300.0, 450.0, 312.0, "监事会")]
    labels = orgchart_parser.merge_words_to_labels(words, gap=5.0)
    # 用集合比较：本条只关心「两个远字块没被并成一个」，不关心标签顺序
    # （原稿写的是 sorted(...) == ["股东大会", "监事会"]，但 sorted 结果按码位
    #   是 ["监事会", "股东大会"]（监 U+76D1 < 股 U+80A1），该断言恒假）
    assert {t for t, _ in labels} == {"股东大会", "监事会"}


# --- 建树 -------------------------------------------------------------------
@pytest.fixture(scope="module")
def chart():
    return orgchart_parser.parse_orgchart(PDF, config.ORGCHART_PAGE_IDX)


def test_chart_node_and_edge_counts(chart):
    assert chart.verify["n_nodes"] == 29, "图上实际 29 个框"
    assert chart.verify["n_edges"] == 28
    assert chart.verify["single_root"] is True
    assert chart.root == "股东大会"


def test_tree_shape(chart):
    n = chart.nodes
    assert set(n["股东大会"].children) == {"董事会", "监事会"}
    assert set(n["董事会"].children) == {
        "薪酬与考核委员会", "提名委员会", "战略委员会", "审计委员会",
        "总经理", "董事会秘书"}
    assert set(n["总经理"].children) == {
        "财务部", "市场开发部", "信息系统部", "销售部", "行政人事部",
        "物流部", "上海分公司"}
    assert set(n["市场开发部"].children) == {"研发中心"}
    assert set(n["销售部"].children) == {
        "电话及网络销售部", "渠道销售部", "大客户销售部", "国际贸易部"}
    assert set(n["大客户销售部"].children) == {
        "珠海销售处", "深圳销售处", "北京销售处", "武汉销售处",
        "广州销售处", "成都销售处"}
    assert set(n["董事会秘书"].children) == {"证券部"}
    assert set(n["审计委员会"].children) == {"内部审计部"}


# --- 聚合（Q5 的核心）-------------------------------------------------------
def test_descendants_of_sales_dept(chart):
    assert set(orgchart_parser.descendants(chart, "销售部", depth=1)) == {
        "电话及网络销售部", "渠道销售部", "大客户销售部", "国际贸易部"}


def test_sales_office_count_per_dept(chart):
    """Q5 的答案：哪个销售部的销售处最多 —— 大客户销售部，6 个。"""
    counts = {
        d: len([x for x in orgchart_parser.descendants(chart, d, depth=1)
                if x.endswith("销售处")])
        for d in orgchart_parser.descendants(chart, "销售部", depth=1)
    }
    assert counts["大客户销售部"] == 6
    assert max(counts, key=counts.get) == "大客户销售部"
    assert set(x for x in orgchart_parser.descendants(chart, "大客户销售部", depth=1)) == {
        "珠海销售处", "深圳销售处", "北京销售处", "武汉销售处",
        "广州销售处", "成都销售处"}


def test_render_text_contains_paths(chart):
    txt = orgchart_parser.render_text(chart)
    assert "销售部/大客户销售部" in txt
    assert "成都销售处" in txt


# --- 宁可失败也不产出错树 ----------------------------------------------------
@pytest.mark.parametrize("verify", [
    {"n_nodes": 28, "n_edges": 28, "single_root": True, "roots": ["股东大会"]},
    {"n_nodes": 29, "n_edges": 27, "single_root": True, "roots": ["股东大会"]},
    {"n_nodes": 29, "n_edges": 28, "single_root": False, "roots": ["a", "b"]},
])
def test_assert_expected_raises_on_wrong_shape(verify):
    """节点/边/单根任一项对不上都必须抛错，而不是把错树交给下游去答。"""
    with pytest.raises(RuntimeError):
        orgchart_parser.assert_expected(verify)
