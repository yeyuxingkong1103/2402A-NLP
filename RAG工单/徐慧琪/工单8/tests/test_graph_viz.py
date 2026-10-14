# 工单编号：人工智能NLP-RAG-基于GraphRAG实现金融问答
import os
import tempfile

from src.graph_viz import (filter_subgraph, network_available, to_matplotlib_png,
                           to_vis_html)
from src.kg_schema import Entity, Relation
from src.kg_store import KnowledgeGraph


def _kg():
    kg = KnowledgeGraph()
    kg.add_entity(Entity("Company:平安银行", "平安银行", "Company"))
    kg.add_entity(Entity("Company:招商银行", "招商银行", "Company"))
    kg.add_entity(Entity("Metric:不良贷款率", "不良贷款率(1.65%)", "Metric",
                         attrs={"metric": "不良贷款率", "value": 1.65, "unit": "%"}))
    kg.add_relation(Relation("Company:平安银行", "HAS_METRIC", "Metric:不良贷款率",
                             evidence="不良贷款率1.65%", source_doc="a.pdf", page=23))
    return kg


def test_vis_html_contains_visjs_and_data():
    html = to_vis_html([{"id": "a", "label": "平安银行", "type": "Company",
                         "color": "#2E6FD9"}],
                       [{"from": "a", "to": "a", "label": "HAS_METRIC"}])
    assert "vis-network" in html
    assert "平安银行" in html
    assert "HAS_METRIC" in html
    assert html.strip().startswith("<!DOCTYPE html>") or html.strip().startswith("<html")


def test_vis_html_escapes_script_breakout():
    """节点名里的尖括号/引号不能破坏 HTML（防 XSS 与渲染错乱）。"""
    html = to_vis_html([{"id": "x", "label": "</script><b>注入</b>",
                         "type": "Company", "color": "#000"}], [])
    assert "</script><b>" not in html
    assert "\\u003c" in html or "&lt;" in html


def test_vis_html_has_legend_for_all_types():
    html = to_vis_html([], [])
    for t in ("Company", "Metric", "Department", "Person"):
        assert t in html


def test_filter_subgraph_by_company():
    nodes, edges = filter_subgraph(_kg(), companies=["平安银行"], hops=1)
    labels = {n["label"] for n in nodes}
    assert "平安银行" in labels
    assert edges, "应包含 HAS_METRIC 边"


def test_filter_subgraph_respects_max_nodes():
    nodes, _e = filter_subgraph(_kg(), hops=2, max_nodes=1)
    assert len(nodes) <= 1


def test_matplotlib_png_written():
    nodes, edges = filter_subgraph(_kg(), companies=["平安银行"], hops=1)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "g.png")
        to_matplotlib_png(nodes, edges, p, dpi=60)
        assert os.path.isfile(p)
        assert os.path.getsize(p) > 1000


def test_network_available_returns_bool():
    assert isinstance(network_available(timeout=1.0), bool)
