# 工单编号：人工智能NLP-RAG-基于GraphRAG实现金融问答
import pytest

from src import graph_qa
from src.kg_schema import Entity, Relation
from src.kg_store import KnowledgeGraph


def _kg():
    kg = KnowledgeGraph()
    for name in ("平安银行", "招商银行", "邮储银行"):
        kg.add_entity(Entity(f"Company:{name}", name, "Company"))
    data = [("平安银行", 1.65, "a.pdf", 23), ("招商银行", 1.16, "b.pdf", 19),
            ("邮储银行", 0.86, "c.pdf", 22)]
    for i, (name, val, doc, page) in enumerate(data):
        mid = f"Metric:npl{i}"
        kg.add_entity(Entity(mid, f"不良贷款率({val}%)@2019年末", "Metric",
                             attrs={"metric": "不良贷款率", "value": val,
                                    "unit": "%", "period": "2019年末",
                                    "source_doc": doc, "page": page}))
        kg.add_relation(Relation(f"Company:{name}", "HAS_METRIC", mid,
                                 evidence=f"不良贷款率{val}%",
                                 source_doc=doc, page=page))
    return kg


@pytest.fixture
def patched(monkeypatch):
    monkeypatch.setattr(graph_qa, "_retrieve_text",
                        lambda q, strategy, top_k: [
                            {"chunk_id": "c1", "text": "不良贷款率保持稳健",
                             "source": "a.pdf", "page_idx": 23, "score": 0.9}])
    monkeypatch.setattr(
        graph_qa, "_call_llm",
        lambda messages, **kw: "## 摘要\n三家银行资产质量稳健。\n"
                               "## 一、数据概览\n| 公司 | 指标 | 数值 | 来源 |\n"
                               "|---|---|---|---|\n| 平安银行 | 不良贷款率 | 1.65% | a.pdf 第23页 |\n"
                               "## 二、分项分析\n...\n## 三、趋势与对比\n...\n"
                               "## 四、风险提示\n...\n## 五、数据来源\na.pdf 第23页")
    return None


def test_report_has_title_and_markdown(patched):
    r = graph_qa.generate_report("银行板块资产质量对比", kg=_kg())
    assert r.title
    assert "## 摘要" in r.markdown


def test_report_source_table_covers_all_companies(patched):
    r = graph_qa.generate_report("银行板块资产质量对比", kg=_kg())
    companies = {row["company"] for row in r.source_table}
    assert {"平安银行", "招商银行", "邮储银行"} <= companies


def test_report_source_table_has_pages(patched):
    r = graph_qa.generate_report("银行板块资产质量对比", kg=_kg())
    for row in r.source_table:
        assert row["source_doc"], "每个数字必须可回溯到文档"
        assert row["page"] >= 0, "每个数字必须可回溯到页码"


def test_report_sections_parsed(patched):
    r = graph_qa.generate_report("银行板块资产质量对比", kg=_kg())
    names = [s["name"] for s in r.sections]
    assert any("摘要" in n for n in names)
    assert len(r.sections) >= 3


def test_report_to_dict_json_serializable(patched):
    import json
    r = graph_qa.generate_report("银行板块资产质量对比", kg=_kg())
    json.dumps(r.to_dict(), ensure_ascii=False)


def test_report_injects_deterministic_data_section(monkeypatch):
    """LLM 未按结构输出数据表时，补入图谱取数的确定性数据章节。

    实测：qwen2.5:3b 会自由发挥章节（### / ####），且不引用图谱数据表；
    若不补齐，「每个数字可回溯」的硬性诉求落空。
    """
    monkeypatch.setattr(graph_qa, "_retrieve_text", lambda q, s, t: [])
    monkeypatch.setattr(graph_qa, "_call_llm",
                        lambda messages, **kw: "# 某银行分析报告\n\n### 分项分析\n正文……")
    r = graph_qa.generate_report("银行板块资产质量对比", kg=_kg())
    assert "数据概览" in r.markdown, "必须补入确定性数据章节"
    assert "1.65" in r.markdown, "数据表须含图谱取值"
    assert any("数据概览" in s["name"] for s in r.sections)
    assert r.markdown.index("数据概览") < r.markdown.index("分项分析"), \
        "数据章节应排在模型正文之前（标题行之后）"


def test_report_keeps_llm_data_section_when_present(patched):
    """LLM 已按结构输出「数据概览」时不重复注入。"""
    r = graph_qa.generate_report("银行板块资产质量对比", kg=_kg())
    assert r.markdown.count("数据概览") == 1


def test_report_degrades_without_llm(monkeypatch):
    monkeypatch.setattr(graph_qa, "_retrieve_text", lambda q, s, t: [])
    def boom(_m, **_k):
        raise RuntimeError("LLM 挂了")
    monkeypatch.setattr(graph_qa, "_call_llm", boom)

    r = graph_qa.generate_report("银行板块资产质量对比", kg=_kg())
    assert r.markdown, "降级也必须产出数据部分"
    assert r.degraded is True
    assert r.source_table, "降级时来源表仍应可用"
