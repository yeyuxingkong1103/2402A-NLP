# 工单编号：人工智能NLP-RAG-基于GraphRAG实现金融问答
import io
import json
import os

from src.kg_build import build_layer1

CHUNKS = [
    {"chunk_id": "c1", "heading_path": ["主要会计数据"], "block_type": "text",
     "text": "平安银行2019年末不良贷款率为1.65%，较上年末下降0.10个百分点。",
     "source": "2020-02-14__平安银行__000001__2019年度报告.pdf", "page_idx": 23},
    {"chunk_id": "c2", "heading_path": ["主要会计数据"], "block_type": "text",
     "text": "招商银行2019年末不良贷款率为1.16%，较上年末下降0.20个百分点。",
     "source": "2020-03-21__招商银行__600036__2019年度报告.pdf", "page_idx": 19},
    {"chunk_id": "c3", "heading_path": [], "block_type": "text",
     "text": "目录", "source": "x.pdf", "page_idx": 0},
]


def test_build_layer1_creates_companies_and_metrics():
    kg = build_layer1(CHUNKS, verbose=False)
    assert kg.has_entity("Company:平安银行")
    assert kg.has_entity("Company:招商银行")
    assert kg.nodes_of_type("Metric")


def test_build_layer1_multi_doc_fusion_single_company_node():
    """跨文档：同一公司只应有一个节点。"""
    kg = build_layer1(CHUNKS, verbose=False)
    assert len(kg.nodes_of_type("Company")) == 2


def test_build_layer1_metric_compare_works():
    kg = build_layer1(CHUNKS, verbose=False)
    rows = kg.metric_compare("不良贷款率")
    assert len(rows) == 2
    assert {r["company"] for r in rows} == {"平安银行", "招商银行"}


def test_build_layer1_all_edges_have_provenance():
    kg = build_layer1(CHUNKS, verbose=False)
    for _u, _v, d in kg.g.edges(data=True):
        assert d["source_doc"], "每条边必须有来源文档"
        assert int(d["page"]) >= 0, "每条边必须有页码"


def test_build_layer1_is_idempotent():
    """重复构建不应产生重复边（可重复运行）。"""
    a = build_layer1(CHUNKS, verbose=False)
    b = build_layer1(CHUNKS, verbose=False)
    assert a.g.number_of_edges() == b.g.number_of_edges()


def test_build_layer2_merges_llm_entities_and_cross_layer_edges(monkeypatch):
    """第2层：LLM 抽到的「老实体(已在图) → 新实体」边必须能入图。

    回归点：build_layer2 必须把图中已有节点并入 merged，否则 remap_relations
    会因端点不在 merged 里把这类跨层边全部丢弃。
    """
    from src import kg_build, kg_extract_llm
    from src.kg_schema import Entity, Relation

    kg = build_layer1(CHUNKS, verbose=False)      # 图中已有 Company:平安银行
    ents = [Entity(entity_id="Company:平安银行", name="平安银行", type="Company"),
            Entity(entity_id="Company:平安理财", name="平安理财", type="Company")]
    rels = [Relation(head_id="Company:平安银行", rel_type="CONTROLS",
                     tail_id="Company:平安理财", evidence="平安银行控股平安理财",
                     source_doc="x.pdf", page=1, confidence=0.9)]

    def fake_extract(selected, **kwargs):
        return {"entities": ents, "relations": rels,
                "n_ok": 1, "n_fail": 0, "n_skipped": 0}

    monkeypatch.setattr(kg_extract_llm, "extract_llm", fake_extract)
    out = kg_build.build_layer2(kg, CHUNKS, resume=False, verbose=False)

    assert kg.has_entity("Company:平安理财")
    assert kg.g.has_edge("Company:平安银行", "Company:平安理财")
    assert out["n_ok"] == 1
    assert out["added_edges"] >= 1
