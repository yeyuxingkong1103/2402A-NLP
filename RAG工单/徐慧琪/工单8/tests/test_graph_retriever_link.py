# 工单编号：人工智能NLP-RAG-基于GraphRAG实现金融问答
from src.graph_retriever import EntityLinkCache, link_entities
from src.kg_schema import Entity
from src.kg_store import KnowledgeGraph


def _kg():
    kg = KnowledgeGraph()
    for n, c in (("平安银行", "000001"), ("招商银行", "600036"),
                 ("中国平安", "601318")):
        kg.add_entity(Entity(f"Company:{n}", n, "Company",
                             aliases=[n + "股份有限公司"], attrs={"code": c}))
    return kg


def test_link_company_by_alias():
    kg = _kg()
    got = link_entities("平安银行2019年末不良贷款率是多少？", kg)
    ids = {g["entity_id"] for g in got}
    assert "Company:平安银行" in ids


def test_link_company_by_full_name():
    kg = _kg()
    got = link_entities("招商银行股份有限公司2019年营业收入", kg)
    assert {g["entity_id"] for g in got} == {"Company:招商银行"}


def test_link_prefers_longest_surface():
    """「中国平安」与「平安银行」同时出现时，各自独立链接，不误配。"""
    kg = _kg()
    got = link_entities("对比中国平安和平安银行2019年的净利润", kg)
    ids = {g["entity_id"] for g in got}
    assert ids == {"Company:中国平安", "Company:平安银行"}


def test_link_returns_empty_for_unrelated_question():
    kg = _kg()
    assert link_entities("今天天气怎么样？", kg) == []


def test_link_scores_are_sorted_desc():
    kg = _kg()
    got = link_entities("招商银行和招商证券的对比", kg)
    scores = [g["score"] for g in got]
    assert scores == sorted(scores, reverse=True)


def test_link_includes_surface_span():
    kg = _kg()
    got = link_entities("平安银行不良贷款率", kg)
    top = got[0]
    assert "surface" in top and top["surface"] in "平安银行不良贷款率"


def test_find_companies_en_matches_english_name():
    """英文提问要能命中公司（多语言支持硬性要求）。"""
    from src.kg_dict import find_companies_en
    hits = find_companies_en(
        "What were Postal Savings Bank of China's NPL ratio in 2019?")
    assert [c for c, _s, _e in hits] == ["邮储银行"]


def test_find_companies_en_is_case_insensitive():
    from src.kg_dict import find_companies_en
    assert [c for c, _s, _e in find_companies_en("citic securities revenue")] \
        == ["中信证券"]


def test_find_companies_en_returns_empty_for_chinese_only():
    from src.kg_dict import find_companies_en
    assert find_companies_en("邮储银行2019年不良贷款率") == []


def test_link_company_from_english_question():
    """英文问题把「China Merchants Bank」链接到图谱里的招商银行。"""
    kg = _kg()
    got = link_entities("What was China Merchants Bank's revenue in 2019?", kg)
    assert "Company:招商银行" in {g["entity_id"] for g in got}


def test_link_english_question_prefers_right_company():
    """Postal Savings / Ping An 两个英文名不互相误配。"""
    kg = _kg()
    kg.add_entity(Entity("Company:邮储银行", "邮储银行", "Company",
                         aliases=["中国邮政储蓄银行"], attrs={"code": "601658"}))
    got = link_entities("Compare Ping An Bank and Postal Savings Bank of China", kg)
    ids = {g["entity_id"] for g in got}
    assert ids == {"Company:平安银行", "Company:邮储银行"}


def test_entity_link_cache_hit():
    cache = EntityLinkCache(maxsize=8)
    kg = _kg()
    q = "平安银行不良贷款率"
    first = link_entities(q, kg)
    cache.put(q, first)
    assert cache.get(q) == first
    assert cache.get("未缓存的问题") is None


def test_entity_link_cache_evicts():
    cache = EntityLinkCache(maxsize=2)
    cache.put("a", [1]); cache.put("b", [2]); cache.put("c", [3])
    assert cache.get("a") is None
    assert cache.get("c") == [3]
