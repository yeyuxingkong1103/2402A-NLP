# 工单编号：人工智能NLP-RAG-基于GraphRAG实现金融问答
import pytest

from src.kg_schema import (ENTITY_COLORS, ENTITY_TYPES, RELATION_TYPES, Entity,
                           Relation, make_entity_id, node_color,
                           validate_entity, validate_relation)


def test_ten_entity_types():
    assert len(ENTITY_TYPES) == 10
    for t in ("Company", "Department", "Person", "Institution", "Industry",
              "Business", "Metric", "Event", "Time", "Location"):
        assert t in ENTITY_TYPES


def test_eleven_relation_types():
    assert len(RELATION_TYPES) == 11
    for t in ("HAS_METRIC", "REPORTED_IN", "BELONGS_TO", "IN_INDUSTRY",
              "CONTROLS", "SHAREHOLDER_OF", "SERVES_AS", "OPERATES",
              "COMPETES_WITH", "INVESTS_IN", "ISSUES"):
        assert t in RELATION_TYPES


def test_relation_endpoints_are_valid_types():
    for name, spec in RELATION_TYPES.items():
        for h in spec["head"]:
            assert h in ENTITY_TYPES, f"{name} 的 head 类型 {h} 未定义"
        for t in spec["tail"]:
            assert t in ENTITY_TYPES, f"{name} 的 tail 类型 {t} 未定义"


def test_every_type_has_color():
    for t in ENTITY_TYPES:
        assert node_color(t).startswith("#")
        assert len(node_color(t)) == 7


def test_make_entity_id_is_stable():
    assert make_entity_id("Company", "平安银行") == "Company:平安银行"
    assert make_entity_id("Company", " 平安银行 ") == "Company:平安银行"


def test_validate_entity_ok():
    e = Entity("Company:平安银行", "平安银行", "Company")
    assert validate_entity(e) == []


def test_validate_entity_rejects_unknown_type():
    e = Entity("X:甲", "甲", "NotAType")
    assert any("未知实体类型" in m for m in validate_entity(e))


def test_validate_entity_rejects_empty_name():
    e = Entity("Company:", "", "Company")
    assert any("名称为空" in m for m in validate_entity(e))


def test_validate_relation_ok():
    r = Relation("Company:平安银行", "HAS_METRIC", "Metric:不良贷款率")
    assert validate_relation(r) == []


def test_validate_relation_rejects_unknown_type():
    r = Relation("Company:甲", "NOT_A_REL", "Company:乙")
    assert any("未知关系类型" in m for m in validate_relation(r))


def test_validate_relation_requires_evidence_for_llm_edges():
    """confidence < 1.0 的边（LLM 抽取）必须带 evidence，防幻觉。"""
    r = Relation("Company:甲", "CONTROLS", "Company:乙", confidence=0.8)
    assert any("evidence" in m for m in validate_relation(r))
