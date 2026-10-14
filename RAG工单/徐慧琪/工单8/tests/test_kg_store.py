# 工单编号：人工智能NLP-RAG-基于GraphRAG实现金融问答
import os
import tempfile

import pytest

from src.kg_schema import Entity, Relation
from src.kg_store import KnowledgeGraph


def _kg():
    kg = KnowledgeGraph()
    kg.add_entity(Entity("Company:平安银行", "平安银行", "Company",
                         aliases=["平安银行股份有限公司"],
                         attrs={"code": "000001", "industry": "银行"}))
    kg.add_entity(Entity("Company:招商银行", "招商银行", "Company",
                         attrs={"code": "600036", "industry": "银行"}))
    kg.add_entity(Entity("Metric:不良贷款率(1.65%)@2019年末",
                         "不良贷款率(1.65%)@2019年末", "Metric",
                         attrs={"metric": "不良贷款率", "value": 1.65,
                                "unit": "%", "period": "2019年末",
                                "industry": "银行", "page": 23,
                                "source_doc": "a.pdf"}))
    kg.add_entity(Entity("Metric:不良贷款率(1.16%)@2019年末",
                         "不良贷款率(1.16%)@2019年末", "Metric",
                         attrs={"metric": "不良贷款率", "value": 1.16,
                                "unit": "%", "period": "2019年末",
                                "industry": "银行", "page": 19,
                                "source_doc": "b.pdf"}))
    kg.add_relation(Relation("Company:平安银行", "HAS_METRIC",
                             "Metric:不良贷款率(1.65%)@2019年末",
                             evidence="不良贷款率1.65%", source_doc="a.pdf", page=23))
    kg.add_relation(Relation("Company:招商银行", "HAS_METRIC",
                             "Metric:不良贷款率(1.16%)@2019年末",
                             evidence="不良贷款率1.16%", source_doc="b.pdf", page=19))
    return kg


def test_add_and_get_entity():
    kg = _kg()
    assert kg.has_entity("Company:平安银行")
    e = kg.get_entity("Company:平安银行")
    assert e.attrs["code"] == "000001"
    assert kg.get_entity("Company:不存在") is None


def test_add_entity_rejects_invalid():
    kg = KnowledgeGraph()
    with pytest.raises(ValueError):
        kg.add_entity(Entity("X:甲", "甲", "NotAType"))


def test_add_relation_rejects_invalid_endpoint():
    kg = _kg()
    assert kg.add_relation(Relation("Company:平安银行", "HAS_METRIC",
                                    "Metric:不存在")) is False


def test_add_relation_dedupes_identical_edge():
    kg = _kg()
    before = kg.g.number_of_edges()
    again = kg.add_relation(Relation("Company:平安银行", "HAS_METRIC",
                                     "Metric:不良贷款率(1.65%)@2019年末",
                                     evidence="不良贷款率1.65%",
                                     source_doc="a.pdf", page=23))
    assert again is False
    assert kg.g.number_of_edges() == before


def test_find_by_alias():
    kg = _kg()
    assert "Company:平安银行" in kg.find_by_alias("平安银行股份有限公司")


def test_nodes_of_type():
    kg = _kg()
    assert set(kg.nodes_of_type("Company")) == {"Company:平安银行", "Company:招商银行"}
    assert len(kg.nodes_of_type("Metric")) == 2


def test_neighbors_hop_limit():
    kg = _kg()
    # 在平安银行的指标节点后再接一跳（Metric -REPORTED_IN-> Time），
    # 使 1 跳与 2 跳的结果存在可观察差异；否则两家公司互不连通，
    # hops 断言恒真，测不出截断是否生效。
    kg.add_entity(Entity("Time:2019年末", "2019年末", "Time"))
    kg.add_relation(Relation("Metric:不良贷款率(1.65%)@2019年末", "REPORTED_IN",
                             "Time:2019年末", evidence="2019年末",
                             source_doc="a.pdf", page=23))
    one_hop = kg.neighbors("Company:平安银行", hops=1)
    assert one_hop.has_entity("Metric:不良贷款率(1.65%)@2019年末")
    assert not one_hop.has_entity("Time:2019年末"), "1跳不应越过指标节点"
    assert not one_hop.has_entity("Company:招商银行"), "1跳不应跨到另一公司"
    two_hop = kg.neighbors("Company:平安银行", hops=2)
    assert two_hop.has_entity("Time:2019年末"), "2跳应到达指标下游的时点节点"


def test_neighbors_respects_max_nodes():
    kg = _kg()
    sub = kg.neighbors("Company:平安银行", hops=2, max_nodes=1)
    assert sub.g.number_of_nodes() <= 1


def test_shortest_path():
    kg = _kg()
    kg.add_entity(Entity("Company:某控股", "某控股", "Company"))
    kg.add_relation(Relation("Company:某控股", "CONTROLS", "Company:平安银行"))
    path = kg.shortest_path("Company:某控股", "Company:平安银行")
    assert path == ["Company:某控股", "Company:平安银行"]
    assert kg.shortest_path("Company:某控股", "Company:招商银行", max_hops=1) is None


def test_metric_compare_across_companies():
    kg = _kg()
    rows = kg.metric_compare("不良贷款率", period="2019年末")
    assert len(rows) == 2
    vals = {r["company"]: r["value"] for r in rows}
    assert vals["平安银行"] == 1.65
    assert vals["招商银行"] == 1.16
    assert all(r["page"] for r in rows), "每条结果必须带页码（provenance）"


def test_subtree_for_org_chart():
    kg = KnowledgeGraph()
    kg.add_entity(Entity("Company:邮储银行", "邮储银行", "Company"))
    kg.add_entity(Entity("Department:个人金融板块", "个人金融板块", "Department"))
    kg.add_entity(Entity("Department:信用卡中心", "信用卡中心", "Department"))
    kg.add_relation(Relation("Department:个人金融板块", "BELONGS_TO",
                             "Company:邮储银行"))
    kg.add_relation(Relation("Department:信用卡中心", "BELONGS_TO",
                             "Department:个人金融板块"))
    tree = kg.subtree("Company:邮储银行", "BELONGS_TO")
    assert tree.has_entity("Department:个人金融板块")
    assert tree.has_entity("Department:信用卡中心")


def test_stats():
    kg = _kg()
    st = kg.stats()
    assert st["n_nodes"] == 4
    assert st["n_edges"] == 2
    assert st["by_type"]["Company"] == 2


def test_save_and_load_roundtrip():
    kg = _kg()
    with tempfile.TemporaryDirectory() as d:
        pkl = os.path.join(d, "kg.pkl")
        gml = os.path.join(d, "kg.graphml")
        kg.save(path_pickle=pkl, path_graphml=gml)
        assert os.path.isfile(pkl) and os.path.isfile(gml)
        back = KnowledgeGraph.load(path_pickle=pkl)
        assert back.g.number_of_nodes() == kg.g.number_of_nodes()
        assert back.g.number_of_edges() == kg.g.number_of_edges()
        e = back.get_entity("Company:平安银行")
        assert e.attrs["code"] == "000001"


def test_merge():
    a = _kg()
    b = KnowledgeGraph()
    b.add_entity(Entity("Company:邮储银行", "邮储银行", "Company"))
    a.merge(b)
    assert a.has_entity("Company:邮储银行")
