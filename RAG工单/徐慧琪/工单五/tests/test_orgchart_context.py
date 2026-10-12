# 工单编号：人工智能NLP-RAG-Query理解优化
#          （工单 PDF 原文：人工智能NLP-RAG-Query 理解优化任务）
"""组织结构图入库块（build_chunks）与检索侧注入（context）的测试。

覆盖**本任务自身**的表面：
  · 块记录：1 整树块 + 29 节点块、元数据（source/page_idx/block_type/org_path）、
    父子指向（节点块 → 整树块）、节点块文本含下级/聚合结论/页码；
  · 检索侧：命中 orgchart 块时 expand_parents 注入「整树 + 命中节点的聚合结论」，
    且与其它块**共用** max_chars 预算；citation_map 给出节点路径。

「哪个销售部的销售处最多」这一聚合事实本身由
tests/test_orgchart_parser.py::test_sales_office_count_per_dept 钉住，
此处不重复断言（P9）。
"""
import os

import pytest

from src import config, orgchart_parser
from src import context as ctx_mod

PDF = os.path.join(config.ROOT, "招股说明书2.pdf")
pytestmark = pytest.mark.slow          # 需要读真实 PDF（排除于 -m "not slow"，P9）

SALES_OFFICES = ["珠海销售处", "深圳销售处", "北京销售处",
                 "武汉销售处", "广州销售处", "成都销售处"]


@pytest.fixture(scope="module")
def chart():
    return orgchart_parser.parse_orgchart(PDF, config.ORGCHART_PAGE_IDX)


@pytest.fixture(scope="module")
def chunks(chart):
    return orgchart_parser.build_chunks(chart)


def _node(chunks: list[dict], name: str) -> dict:
    return next(c for c in chunks if c.get("org_node") == name)


# ---------------------------------------------------------------------------
# 一、块记录（build_chunks）
# ---------------------------------------------------------------------------
def test_build_chunks_one_tree_plus_one_block_per_node(chart, chunks):
    tree = [c for c in chunks if c.get("org_node") == ""]
    nodes = [c for c in chunks if c.get("org_node")]
    assert {c["block_type"] for c in chunks} == {"orgchart"}
    assert len(tree) == 1, "整树父块有且只有一个"
    assert len(nodes) == 29, "一个节点一块"
    assert {c["org_node"] for c in nodes} == set(chart.nodes)
    assert len({c["chunk_id"] for c in chunks}) == len(chunks), "chunk_id 不得重复"


def test_node_blocks_point_to_the_tree_block(chunks):
    tree_id = chunks[0]["chunk_id"]
    assert tree_id.endswith(f"p{config.ORGCHART_PAGE_IDX}#orgchart")
    for c in chunks:
        assert c["parent_id"] == tree_id
        assert c["org_tree_id"] == tree_id


def test_node_chunk_carries_required_metadata(chunks):
    c = _node(chunks, "大客户销售部")
    assert c["source"] == config.ORGCHART_SOURCE
    assert c["page_idx"] == config.ORGCHART_PAGE_IDX
    assert c["block_type"] == "orgchart"
    # org_path 取 path_of 的真实输出（自根到本节点的完整路径，与 render_text
    # 行尾的「← 路径：…」同口径），不是从某个中间节点起的相对路径。
    assert c["org_path"] == "股东大会/董事会/总经理/销售部/大客户销售部"
    assert c["org_path"].endswith("销售部/大客户销售部")
    assert c["parent_id"] == c["org_tree_id"]
    assert "销售处" in c["text"]


def test_node_chunk_text_has_children_aggregation_and_page(chunks):
    """节点块文本必须自带下级清单与聚合结论（生成期直接可用）。"""
    c = _node(chunks, "大客户销售部")
    for name in SALES_OFFICES:
        assert name in c["text"]
    assert "下级6个销售处" in c["text"], "聚合结论要写进块文本"
    assert str(config.ORGCHART_PAGE_IDX + 1) in c["text"], "页码用展示口径"

    leaf = _node(chunks, "成都销售处")
    assert "无下级" in leaf["text"]
    assert leaf["org_path"] == "股东大会/董事会/总经理/销售部/大客户销售部/成都销售处"


def test_heading_path_is_a_list(chunks):
    """heading_path 必须是列表：入库口径（vector_store._to_point 走 list()）
    与上下文展示（build_context 走 " / ".join）都按列表消费，字符串会被逐字拆开。"""
    for c in chunks:
        assert isinstance(c["heading_path"], list) and c["heading_path"]


def test_tree_chunk_holds_whole_tree(chunks):
    tree = next(c for c in chunks if c.get("org_node") == "")
    assert "武汉力源信息技术股份有限公司组织结构图" in tree["text"]
    assert "股东大会" in tree["parent_text"] and "成都销售处" in tree["parent_text"]
    assert "组织结构图" in " / ".join(tree["heading_path"])


# ---------------------------------------------------------------------------
# 二、检索侧注入（expand_parents / citation_map）
# ---------------------------------------------------------------------------
def test_context_injects_subtree_and_aggregation(chunks):
    hit = dict(_node(chunks, "销售部"), score=1.0)
    blocks = ctx_mod.expand_parents([hit], max_chars=4000)
    assert blocks, "命中 orgchart 节点必须能展开"
    b = blocks[0]
    assert b["block_type"] == "orgchart"
    assert b["org_path"] == "股东大会/董事会/总经理/销售部"
    assert b["page_idx"] == config.ORGCHART_PAGE_IDX
    text = b["parent_text"]
    assert "大客户销售部" in text
    assert "珠海销售处" in text, "父块必须是整树/子树，不是只给一个节点名"


def test_context_keeps_hit_node_aggregation_line(chunks):
    """命中节点块的「路径 + 下级 + 统计」必须进上下文——聚合题的证据在这里。"""
    hit = dict(_node(chunks, "大客户销售部"), score=1.0)
    b = ctx_mod.expand_parents([hit], max_chars=4000)[0]
    text = b["parent_text"]
    assert "下级6个销售处" in text
    for name in SALES_OFFICES:
        assert name in text


def test_context_dedupes_orgchart_hits_into_one_block(chunks):
    """同一张图命中多个节点只注入一份整树（否则整树会在上下文里重复占预算）。"""
    hits = [dict(_node(chunks, "销售部"), score=1.0),
            dict(_node(chunks, "大客户销售部"), score=0.9)]
    blocks = ctx_mod.expand_parents(hits, max_chars=4000)
    org = [b for b in blocks if b.get("block_type") == "orgchart"]
    assert len(org) == 1
    assert "下级6个销售处" in org[0]["parent_text"]


def test_context_tree_hit_is_not_duplicated(chunks):
    """命中整树块本身：给整树即可——其块文本=「表头 + 整树」，不能再叠一遍。"""
    tree = dict(next(c for c in chunks if c.get("org_node") == ""), score=1.0)
    b = ctx_mod.expand_parents([tree], max_chars=4000)[0]
    assert b["parent_text"] == tree["parent_text"]


def test_context_orgchart_shares_budget(chunks):
    """orgchart 块必须与其它块共用 max_chars 预算（不得绕过记账）。"""
    other = {"chunk_id": "c1", "text": "注册资本为7,360万元", "parent_id": "s1",
             "parent_text": "正文。" * 200, "parent_page_idx": 5, "page_idx": 5,
             "heading_path": ["第三节"], "source": "招股说明书1.pdf",
             "block_type": "text", "score": 0.9}
    hit = dict(_node(chunks, "销售部"), score=1.0)
    blocks = ctx_mod.expand_parents([other, hit], max_chars=1500)
    total = sum(len(b["parent_text"]) for b in blocks)
    assert any(b.get("block_type") == "orgchart" for b in blocks), "预算内应装入 orgchart 块"
    assert total <= 1500, f"超出预算：{total}"


def test_citation_map_marks_orgchart_node_path(chunks):
    """引用要带「页 + 组织结构图标题 + 节点路径」，与表格的 table_id/caption 同写法。"""
    hit = dict(_node(chunks, "大客户销售部"), score=1.0)
    cm = ctx_mod.citation_map(ctx_mod.expand_parents([hit], max_chars=4000))
    assert cm[0]["page_display"] == config.ORGCHART_PAGE_IDX + 1
    assert cm[0]["source"] == config.ORGCHART_SOURCE
    assert cm[0]["org_path"].endswith("销售部/大客户销售部")
    assert "组织结构图" in cm[0]["heading_path"][-2]
    ctx = ctx_mod.build_context(ctx_mod.expand_parents([hit], max_chars=4000))
    assert "【组织结构图" in ctx, "组织结构图片段必须在上下文里带标记"
