# 工单编号：人工智能NLP-RAG-基于GraphRAG实现金融问答
import pytest

from src import graph_qa
from src.kg_schema import Entity, Relation
from src.kg_store import KnowledgeGraph


def _kg():
    kg = KnowledgeGraph()
    kg.add_entity(Entity("Company:平安银行", "平安银行", "Company"))
    kg.add_entity(Entity("Metric:不良贷款率", "不良贷款率(1.65%)@2019年末", "Metric",
                         attrs={"metric": "不良贷款率", "value": 1.65, "unit": "%",
                                "period": "2019年末", "source_doc": "a.pdf",
                                "page": 23}))
    kg.add_relation(Relation("Company:平安银行", "HAS_METRIC", "Metric:不良贷款率",
                             evidence="不良贷款率1.65%", source_doc="a.pdf", page=23))
    return kg


@pytest.fixture
def patched(monkeypatch):
    """桩掉检索与 LLM，只测编排逻辑。"""
    monkeypatch.setattr(graph_qa, "_retrieve_text",
                        lambda q, strategy, top_k: [
                            {"chunk_id": "c1", "text": "平安银行不良贷款率1.65%",
                             "source": "a.pdf", "page_idx": 23, "score": 0.9}])
    monkeypatch.setattr(graph_qa, "_call_llm",
                        lambda messages, **kw: "平安银行2019年末不良贷款率为1.65%"
                                               "（a.pdf 第23页）。")
    return None


def test_ask_returns_result_with_all_parts(patched):
    res = graph_qa.ask("平安银行2019年末不良贷款率是多少？", kg=_kg())
    assert res.answer
    assert res.mode in ("khop", "metric_compare", "path", "org_tree")
    assert res.graph_facts
    assert res.subgraph["nodes"]
    assert res.text_hits


def test_ask_records_timings(patched):
    res = graph_qa.ask("平安银行不良贷款率", kg=_kg())
    assert "total" in res.timings
    assert res.timings["total"] >= 0
    for k in ("link", "subgraph", "retrieve", "generate"):
        assert k in res.timings


def test_ask_detects_language(patched):
    assert graph_qa.ask("平安银行不良贷款率", kg=_kg()).lang == "zh"
    assert graph_qa.ask("What is Ping An Bank's NPL ratio?",
                        kg=_kg()).lang == "en"


def test_ask_degrades_when_llm_fails(monkeypatch):
    """LLM 失败时必须降级返回图谱事实与文本块，而非整体报错（容错要求7）。"""
    monkeypatch.setattr(graph_qa, "_retrieve_text",
                        lambda q, strategy, top_k: [])
    def boom(_m, **_k):
        raise RuntimeError("Ollama 不可用")
    monkeypatch.setattr(graph_qa, "_call_llm", boom)

    res = graph_qa.ask("平安银行不良贷款率", kg=_kg())
    assert res.answer, "降级也必须给出答案（抽取式）"
    assert "Ollama" not in res.answer
    res_dict = res.to_dict()
    assert res_dict["degraded"] is True


def test_ask_handles_no_entities(patched):
    res = graph_qa.ask("今天天气怎么样？", kg=_kg())
    assert res.answer
    assert res.mode == "none"


def test_ask_with_dialogue_state(patched):
    from src.dialogue import DialogueState
    st = DialogueState(session_id="t")
    res = graph_qa.ask("平安银行不良贷款率", kg=_kg(), state=st)
    assert res.answer
    assert res.question


def test_result_to_dict_is_json_serializable(patched):
    import json
    res = graph_qa.ask("平安银行不良贷款率", kg=_kg())
    json.dumps(res.to_dict(), ensure_ascii=False)


def test_graph_cache_returns_same_subgraph(patched):
    kg = _kg()
    q = "平安银行不良贷款率"
    a = graph_qa.ask(q, kg=kg)
    b = graph_qa.ask(q, kg=kg)
    assert a.subgraph["nodes"] == b.subgraph["nodes"]


def test_extractive_answer_cites_hit_page():
    """降级答案要引用命中页码：语料命中带 1 基 `page` 且**没有** page_idx，
    只读 page_idx 会输出「第 -1 页」。"""
    from src.graph_qa import _extractive_answer
    from src.graph_retriever import SubgraphResult

    txt = _extractive_answer(SubgraphResult(mode="none"),
                             [{"source": "a.pdf", "page": 39, "text": "原文"}],
                             "zh")
    assert "第 39 页" in txt
    assert "-1" not in txt
