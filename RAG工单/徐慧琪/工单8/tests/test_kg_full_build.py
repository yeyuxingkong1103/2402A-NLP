# 工单编号：人工智能NLP-RAG-基于GraphRAG实现金融问答
"""图谱质量断言（对已构建的真实图谱做检查；图谱不存在时跳过）。"""
import os

import pytest

from src import config

pytestmark = pytest.mark.skipif(
    not os.path.isfile(config.KG_PICKLE),
    reason="图谱未构建，先运行 python build_kg.py --layer1")


def _kg():
    from src.kg_store import KnowledgeGraph
    return KnowledgeGraph.load()


def test_covers_all_nine_companies():
    kg = _kg()
    companies = {kg.g.nodes[n]["name"] for n in kg.nodes_of_type("Company")}
    for c in ("平安银行", "中国平安", "招商银行", "邮储银行", "中信证券",
              "中国人寿", "中国太保", "招商证券", "国泰君安"):
        assert c in companies, f"缺少公司节点：{c}"


def test_covers_all_nine_documents():
    kg = _kg()
    docs = {d.get("source_doc") for _u, _v, d in kg.g.edges(data=True)
            if d.get("source_doc")}
    assert len(docs) >= 9, f"图谱只覆盖 {len(docs)} 份文档"


def test_has_metric_nodes_for_each_industry():
    """银行/保险/证券三类指标都应出现。"""
    kg = _kg()
    metrics = {kg.g.nodes[n]["attrs"].get("metric")
               for n in kg.nodes_of_type("Metric")}
    for m in ("不良贷款率", "拨备覆盖率", "营业收入"):
        assert m in metrics, f"缺少银行类指标：{m}"


def test_all_edges_have_provenance():
    kg = _kg()
    bad = [(_u, _v, d) for _u, _v, d in kg.g.edges(data=True)
           if not d.get("source_doc") or int(d.get("page", -1)) < 0]
    assert not bad, f"{len(bad)} 条边缺少 provenance"


def test_metric_compare_returns_multiple_companies():
    kg = _kg()
    rows = kg.metric_compare("不良贷款率")
    assert len({r["company"] for r in rows}) >= 2
