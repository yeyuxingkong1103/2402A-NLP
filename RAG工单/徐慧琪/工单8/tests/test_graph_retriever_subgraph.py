# 工单编号：人工智能NLP-RAG-基于GraphRAG实现金融问答
from src.graph_retriever import detect_mode, retrieve_subgraph
from src.kg_schema import Entity, Relation
from src.kg_store import KnowledgeGraph


def _kg():
    kg = KnowledgeGraph()
    kg.add_entity(Entity("Company:平安银行", "平安银行", "Company",
                         attrs={"industry": "银行"}))
    kg.add_entity(Entity("Company:招商银行", "招商银行", "Company",
                         attrs={"industry": "银行"}))
    kg.add_entity(Entity("Metric:不良贷款率", "不良贷款率(1.65%)@2019年末",
                         "Metric",
                         attrs={"metric": "不良贷款率", "value": 1.65,
                                "unit": "%", "period": "2019年末",
                                "source_doc": "a.pdf", "page": 23}))
    kg.add_entity(Entity("Metric:不良贷款率2", "不良贷款率(1.16%)@2019年末",
                         "Metric",
                         attrs={"metric": "不良贷款率", "value": 1.16,
                                "unit": "%", "period": "2019年末",
                                "source_doc": "b.pdf", "page": 19}))
    kg.add_relation(Relation("Company:平安银行", "HAS_METRIC",
                             "Metric:不良贷款率", evidence="不良贷款率1.65%",
                             source_doc="a.pdf", page=23))
    kg.add_relation(Relation("Company:招商银行", "HAS_METRIC",
                             "Metric:不良贷款率2", evidence="不良贷款率1.16%",
                             source_doc="b.pdf", page=19))
    return kg


def _linked(name, etype="Company"):
    return [{"entity_id": f"{etype}:{name}", "name": name, "type": etype,
             "score": 0.9, "surface": name}]


def test_detect_mode_metric_compare():
    linked = _linked("平安银行") + _linked("招商银行")
    q = "对比平安银行与招商银行2019年末的不良贷款率"
    assert detect_mode(q, linked) == "metric_compare"


def test_detect_mode_org_tree():
    q = "邮储银行的组织结构图中总行分为哪几个板块？"
    assert detect_mode(q, _linked("邮储银行")) == "org_tree"


def test_detect_mode_path_for_two_entities():
    q = "中国平安和平安银行是什么关系？"
    assert detect_mode(q, _linked("中国平安") + _linked("平安银行")) == "path"


def test_detect_mode_defaults_to_khop():
    assert detect_mode("平安银行的资产质量如何？", _linked("平安银行")) == "khop"


def test_metric_compare_returns_structured_facts():
    kg = _kg()
    linked = _linked("平安银行") + _linked("招商银行")
    sub = retrieve_subgraph("对比平安银行与招商银行的不良贷款率", linked, kg)
    assert sub.mode == "metric_compare"
    assert len(sub.facts) >= 2
    assert any("1.65" in f for f in sub.facts)
    assert any("1.16" in f for f in sub.facts)


def test_facts_carry_source_and_page():
    """每条事实必须可回溯到文本块（工单核心诉求）。"""
    kg = _kg()
    sub = retrieve_subgraph("对比平安银行与招商银行的不良贷款率",
                            _linked("平安银行") + _linked("招商银行"), kg)
    assert sub.source_pages
    for sp in sub.source_pages:
        assert sp["source_doc"] and sp["page"] >= 0


def test_metric_compare_restricts_to_linked_companies():
    """对比题只应给出被链接公司的取值，不得混入未提及公司的数据。

    实测缺陷：问「对比平安银行与招商银行的不良贷款率」时，子图把
    邮储银行 0.86% 也塞进事实区，LLM 据此答成「邮储银行更低」。
    """
    kg = _kg()
    kg.add_entity(Entity("Company:邮储银行", "邮储银行", "Company"))
    kg.add_entity(Entity("Metric:不良贷款率3", "不良贷款率(0.86%)@2019年末",
                         "Metric",
                         attrs={"metric": "不良贷款率", "value": 0.86,
                                "unit": "%", "period": "2019年末",
                                "source_doc": "c.pdf", "page": 21}))
    kg.add_relation(Relation("Company:邮储银行", "HAS_METRIC",
                             "Metric:不良贷款率3", evidence="不良贷款率0.86%",
                             source_doc="c.pdf", page=21))

    sub = retrieve_subgraph("对比平安银行与招商银行的不良贷款率",
                            _linked("平安银行") + _linked("招商银行"), kg)
    assert sub.mode == "metric_compare"
    text = "\n".join(sub.facts)
    assert "平安银行" in text and "招商银行" in text
    assert "邮储银行" not in text, "未链接公司的数据不得进入对比事实"


def test_khop_returns_facts_and_edges():
    kg = _kg()
    sub = retrieve_subgraph("平安银行的不良贷款率", _linked("平安银行"), kg)
    assert sub.mode == "khop"
    assert sub.facts
    assert sub.edges


def test_empty_link_returns_empty_result():
    kg = _kg()
    sub = retrieve_subgraph("无关问题", [], kg)
    assert sub.facts == []
    assert sub.mode == "none"


def test_subgraph_to_html_data_shape():
    kg = _kg()
    sub = retrieve_subgraph("平安银行的不良贷款率", _linked("平安银行"), kg)
    data = sub.to_html_data()
    assert {"nodes", "edges"} <= set(data)
    for n in data["nodes"]:
        assert {"id", "label", "type", "color"} <= set(n)
    for e in data["edges"]:
        assert {"from", "to", "label"} <= set(e)
