# 工单编号：人工智能NLP-RAG-基于GraphRAG实现金融问答
import pytest

from src.kg_prompt import (build_extract_messages, load_prompt,
                           parse_llm_json, to_entities_relations)

CHUNK = {
    "chunk_id": "c1",
    "text": "本公司控股股东为中国平安保险（集团）股份有限公司，持股比例49.56%。",
    "source": "2020-02-21__中国平安__601318__2019年度报告.pdf",
    "page_idx": 42,
    "heading_path": ["公司治理", "股东情况"],
}

GOOD = '''{"entities": [
  {"name": "中国平安保险（集团）股份有限公司", "type": "Company",
   "aliases": ["中国平安"], "attrs": {}},
  {"name": "招商银行股份有限公司", "type": "Company", "aliases": ["招商银行"],
   "attrs": {}}
], "relations": [
  {"head": "中国平安保险（集团）股份有限公司", "type": "CONTROLS",
   "tail": "招商银行股份有限公司", "evidence": "控股股东为中国平安保险（集团）股份有限公司",
   "confidence": 0.9}
]}'''


def test_load_prompt_reads_yaml():
    p = load_prompt("kg_extract")
    assert "system" in p and "user_template" in p
    assert "entity_types" in p["system"]
    assert "relation_types" in p["system"]


def test_build_messages_renders_types_and_text():
    msgs = build_extract_messages(CHUNK)
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    assert "Company" in msgs[0]["content"]
    assert "CONTROLS" in msgs[0]["content"]
    assert "中国平安保险（集团）股份有限公司" in msgs[1]["content"]
    assert "第 42 页" in msgs[1]["content"]
    assert "公司治理" in msgs[1]["content"]


def test_parse_llm_json_plain():
    got = parse_llm_json(GOOD)
    assert len(got["entities"]) == 2


def test_parse_llm_json_strips_markdown_fence():
    got = parse_llm_json(f"```json\n{GOOD}\n```")
    assert len(got["entities"]) == 2


def test_parse_llm_json_recovers_from_trailing_prose():
    got = parse_llm_json(GOOD + "\n\n以上为抽取结果。")
    assert len(got["entities"]) == 2


def test_parse_llm_json_raises_on_garbage():
    with pytest.raises(ValueError):
        parse_llm_json("这不是 JSON")


def test_to_entities_relations_builds_objects():
    ents, rels = to_entities_relations(parse_llm_json(GOOD), CHUNK)
    ids = {e.entity_id for e in ents}
    # 本层保留 LLM 返回的原始名（归一由 kg_dedup.dedup_entities 在建图入库前统一完成）
    assert "Company:中国平安保险（集团）股份有限公司" in ids
    assert "Company:招商银行股份有限公司" in ids
    assert len(rels) == 1
    assert rels[0].rel_type == "CONTROLS"
    assert rels[0].source_doc == CHUNK["source"]
    assert rels[0].page == 42
    assert rels[0].confidence == 0.9
    # 归一后应落到规范简称
    from src.kg_dedup import dedup_entities
    merged = dedup_entities(ents)
    assert "Company:中国平安" in merged


def test_to_entities_relations_drops_relation_without_evidence():
    """无 evidence 的关系必须丢弃（防幻觉硬约束）。"""
    payload = {"entities": [{"name": "甲", "type": "Company", "aliases": []},
                            {"name": "乙", "type": "Company", "aliases": []}],
               "relations": [{"head": "甲", "type": "CONTROLS", "tail": "乙",
                              "evidence": "", "confidence": 0.9}]}
    _e, rels = to_entities_relations(payload, CHUNK)
    assert rels == []


def test_to_entities_relations_drops_relation_with_fabricated_evidence():
    """evidence 未在原文中出现 → 丢弃（防幻觉硬约束）。"""
    payload = {"entities": [{"name": "甲", "type": "Company", "aliases": []},
                            {"name": "乙", "type": "Company", "aliases": []}],
               "relations": [{"head": "甲", "type": "CONTROLS", "tail": "乙",
                              "evidence": "这句话原文里根本没有出现过",
                              "confidence": 0.9}]}
    _e, rels = to_entities_relations(payload, CHUNK)
    assert rels == []


def test_to_entities_relations_drops_unknown_entity_type():
    payload = {"entities": [{"name": "甲", "type": "外星人", "aliases": []}],
               "relations": []}
    ents, _r = to_entities_relations(payload, CHUNK)
    assert ents == []


def test_parse_llm_json_rejects_top_level_array():
    """顶层是 JSON 数组（非对象）→ ValueError，而非调用方 AttributeError。"""
    with pytest.raises(ValueError):
        parse_llm_json("[1, 2, 3]")


def test_to_entities_relations_drops_unknown_relation_type():
    """闸门2：关系类型不在本体中 → 丢弃。"""
    payload = {"entities": [{"name": "甲", "type": "Company", "aliases": []},
                            {"name": "乙", "type": "Company", "aliases": []}],
               "relations": [{"head": "甲", "type": "NOT_A_REL", "tail": "乙",
                              "evidence": "控股股东为中国平安保险（集团）股份有限公司",
                              "confidence": 0.9}]}
    _e, rels = to_entities_relations(payload, CHUNK)
    assert rels == []


def test_to_entities_relations_drops_illegal_head_tail_types():
    """闸门2：HAS_METRIC 要求头 Company → 尾 Metric；Metric 头必须被丢弃。"""
    payload = {"entities": [{"name": "不良贷款率", "type": "Metric", "aliases": []},
                            {"name": "报告期末", "type": "Time", "aliases": []}],
               "relations": [{"head": "不良贷款率", "type": "HAS_METRIC",
                              "tail": "报告期末",
                              "evidence": "控股股东为中国平安保险（集团）股份有限公司",
                              "confidence": 0.9}]}
    _e, rels = to_entities_relations(payload, CHUNK)
    assert rels == []


TRUNCATED = '''{
  "entities": [
    {"name": "中国平安保险（集团）股份有限公司", "type": "Company",
     "aliases": [], "attrs": {}},
    {"name": "招商银行股份有限公司", "type": "Company",
     "aliases": [], "attrs": {}},
    {"name": "未完'''


def test_parse_llm_json_salvages_objects_before_truncation():
    """截断输出（最外层不闭合）→ 抢救已完整的实体对象，不抛异常。"""
    got = parse_llm_json(TRUNCATED)
    assert [e["name"] for e in got["entities"]] == [
        "中国平安保险（集团）股份有限公司", "招商银行股份有限公司"]
    assert got["relations"] == []


def test_salvage_scanner_respects_braces_and_quotes_in_strings():
    """字符串值中的 { } 与转义 \\" 不得干扰括号匹配。"""
    payload = ('{"entities": [\n'
               '  {"name": "含{花括号}与引号\\"的名字", "type": "Company",'
               ' "aliases": [], "attrs": {}},\n'
               '  {"name": "被截断')
    got = parse_llm_json(payload)
    assert [e["name"] for e in got["entities"]] == ['含{花括号}与引号"的名字']


def test_parse_llm_json_truncated_without_salvage_still_raises():
    """截断但无任何可抢救对象 → 仍是 ValueError（契约不变）。"""
    with pytest.raises(ValueError):
        parse_llm_json('{"entities": [{"name": "甲"')


def test_salvage_caps_recovered_objects():
    """退化重复输出（同一对象几十次）下回收有上限，不得跑飞。"""
    degenerate = '{"entities": [' + (
        '{"name": "甲", "type": "Company", "aliases": [], "attrs": {}},'
        * 500) + '{"name": "未完'
    got = parse_llm_json(degenerate)
    assert 0 < len(got["entities"]) <= 200


def test_salvaged_relations_still_pass_all_gates():
    """抢救出的关系仍受闸门约束：证据落地才留，缺失/编造的一律丢。"""
    truncated = (
        '{"entities": ['
        '{"name": "甲", "type": "Company", "aliases": [], "attrs": {}},'
        '{"name": "乙", "type": "Company", "aliases": [], "attrs": {}}],'
        '"relations": ['
        '{"head": "甲", "type": "CONTROLS", "tail": "乙",'
        ' "evidence": "控股股东为中国平安保险（集团）股份有限公司",'
        ' "confidence": 0.9},'
        '{"head": "乙", "type": "CONTROLS", "tail": "甲",'
        ' "evidence": "这句证据是编造的", "confidence": 0.9},'
        '{"head": "甲", "type": "CONTROLS", "tail": "乙", "confidence": 0.9}')
    payload = parse_llm_json(truncated)
    ents, rels = to_entities_relations(payload, CHUNK)
    assert len(ents) == 2
    assert len(rels) == 1                       # 证据缺失 / 编造的两条被闸门3丢弃
    assert rels[0].evidence.startswith("控股股东")


def test_to_entities_relations_bad_confidence_drops_only_that_relation():
    """Fix 1（逐项容错）：第二条 confidence 非数值只丢它，第一条照常保留。"""
    payload = {"entities": [{"name": "甲", "type": "Company", "aliases": []},
                            {"name": "乙", "type": "Company", "aliases": []}],
               "relations": [
                   {"head": "甲", "type": "CONTROLS", "tail": "乙",
                    "evidence": "控股股东为中国平安保险（集团）股份有限公司",
                    "confidence": 0.9},
                   {"head": "乙", "type": "CONTROLS", "tail": "甲",
                    "evidence": "控股股东为中国平安保险（集团）股份有限公司",
                    "confidence": "high"}]}
    _e, rels = to_entities_relations(payload, CHUNK)
    assert len(rels) == 1
    assert rels[0].confidence == 0.9


def test_to_entities_relations_drops_entity_with_non_dict_attrs():
    """Fix 1：attrs 非 dict 的实体只丢该实体，其余实体保留。"""
    payload = {"entities": [{"name": "甲", "type": "Company", "aliases": [],
                             "attrs": "oops"},
                            {"name": "乙", "type": "Company", "aliases": []}],
               "relations": []}
    ents, _r = to_entities_relations(payload, CHUNK)
    assert [e.entity_id for e in ents] == ["Company:乙"]
