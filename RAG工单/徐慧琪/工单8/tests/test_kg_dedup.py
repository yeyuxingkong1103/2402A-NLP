# 工单编号：人工智能NLP-RAG-基于GraphRAG实现金融问答
from src.kg_dedup import (canonical_company, dedup_entities, normalize_name,
                          remap_relations, resolve_id)
from src.kg_schema import Entity, Relation


def test_normalize_strips_and_unifies():
    assert normalize_name(" 平安银行 ", "Company") == "平安银行"
    assert normalize_name("招商银行（集团）", "Company") == "招商银行(集团)"
    assert normalize_name("ＡＢＣ", "Company") == "ABC"


def test_normalize_company_drops_suffix():
    assert normalize_name("平安银行股份有限公司", "Company") == "平安银行"


def test_canonical_company_maps_all_aliases():
    for surface in ("平安银行股份有限公司", "平安银行", "平安银行股份"):
        assert canonical_company(surface) == "平安银行"
    assert canonical_company("中国平安保险（集团）股份有限公司") == "中国平安"


def test_canonical_company_returns_none_for_unknown():
    assert canonical_company("某不存在的银行") is None


def test_resolve_id_prefers_company_canonical():
    assert resolve_id("Company", "招商银行股份有限公司") == "Company:招商银行"


def test_dedup_merges_same_entity_by_alias():
    a = Entity("Company:平安银行", "平安银行", "Company", aliases=["平安银行"])
    b = Entity("Company:平安银行股份有限公司", "平安银行股份有限公司", "Company",
               aliases=["平安银行股份有限公司"])
    merged = dedup_entities([a, b])
    assert len(merged) == 1
    only = next(iter(merged.values()))
    assert only.entity_id == "Company:平安银行"
    assert "平安银行股份有限公司" in only.aliases


def test_dedup_keeps_distinct_entities_separate():
    a = Entity("Company:中国平安", "中国平安", "Company")
    b = Entity("Company:平安银行", "平安银行", "Company")
    merged = dedup_entities([a, b])
    assert len(merged) == 2


def test_dedup_fills_missing_attrs_from_duplicate():
    a = Entity("Metric:不良贷款率(1.65%)@2019年末", "不良贷款率(1.65%)@2019年末",
               "Metric", attrs={"value": 1.65})
    b = Entity("Metric:不良贷款率(1.65%)@2019年末", "不良贷款率(1.65%)@2019年末",
               "Metric", attrs={"unit": "%"})
    merged = dedup_entities([a, b])
    only = next(iter(merged.values()))
    assert only.attrs["value"] == 1.65
    assert only.attrs["unit"] == "%"


def test_remap_relations_rewrites_endpoints():
    src = Entity("Company:平安银行股份有限公司", "平安银行股份有限公司", "Company")
    tgt = Entity("Metric:不良贷款率(1.65%)@2019年末",
                 "不良贷款率(1.65%)@2019年末", "Metric")
    merged = dedup_entities([src, tgt])
    rels = [Relation("Company:平安银行股份有限公司", "HAS_METRIC",
                     "Metric:不良贷款率(1.65%)@2019年末")]
    out = remap_relations(rels, merged)
    assert len(out) == 1
    assert out[0].head_id in merged
    assert out[0].tail_id in merged


def test_remap_relations_drops_orphans():
    """端点已被合并掉的边应被丢弃，而非留下悬空引用。"""
    out = remap_relations([Relation("Company:不存在", "CONTROLS", "Company:也不存在")],
                          {})
    assert out == []
