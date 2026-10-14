# 工单编号：人工智能NLP-RAG-基于GraphRAG实现金融问答
from src.kg_dict import (COMPANY_TABLE, METRIC_TABLE, SECTION_WHITELIST,
                         find_companies, find_institutions, find_metrics)


def test_nine_companies_with_metadata():
    assert len(COMPANY_TABLE) == 9
    for name, meta in COMPANY_TABLE.items():
        assert meta["code"].isdigit() and len(meta["code"]) == 6, name
        assert meta["industry"] in ("银行", "保险", "证券"), name
        assert meta["full"].endswith(("公司", "集团")), name


def test_company_industries_cover_three_sectors():
    inds = {m["industry"] for m in COMPANY_TABLE.values()}
    assert inds == {"银行", "保险", "证券"}


def test_find_companies_by_alias():
    hits = find_companies("平安银行股份有限公司2019年年度报告")
    names = {h[0] for h in hits}
    assert "平安银行" in names


def test_find_companies_span_is_correct():
    text = "招商银行2019年"
    hits = find_companies(text)
    assert hits, "应命中招商银行"
    name, start, end = hits[0]
    assert text[start:end] in ("招商银行", "招商银行股份有限公司")


def test_find_companies_no_false_positive_on_chaoshang():
    """「招商证券」不应被误识别为「招商银行」。"""
    names = {h[0] for h in find_companies("招商证券2021年年度报告")}
    assert "招商证券" in names
    assert "招商银行" not in names


def test_metrics_cover_core_financial_indicators():
    for m in ("不良贷款率", "拨备覆盖率", "营业收入", "归属于母公司股东的净利润",
              "内含价值", "新业务价值", "资本充足率", "总投资收益"):
        assert m in METRIC_TABLE, m


def test_metric_unit_kind():
    assert METRIC_TABLE["不良贷款率"]["unit_kind"] == "percent"
    assert METRIC_TABLE["营业收入"]["unit_kind"] == "money"
    assert METRIC_TABLE["内含价值"]["unit_kind"] == "money"


def test_find_metrics_returns_alias_surface():
    hits = find_metrics("归属于母公司股东的净利润为100亿元")
    canon = {h[0] for h in hits}
    assert "归属于母公司股东的净利润" in canon
    surface = [h[1] for h in hits if h[0] == "归属于母公司股东的净利润"][0]
    assert "净利润" in surface


def test_find_institutions():
    names = {h[0] for h in find_institutions("中国银行保险监督管理委员会发布")}
    assert names, "应命中监管机构"


def test_section_whitelist_has_key_sections():
    joined = "|".join(SECTION_WHITELIST)
    for kw in ("公司治理", "组织结构", "股东", "董事", "监事", "高级管理人员",
               "主要会计数据", "业务概要", "风险管理", "子公司", "关联"):
        assert kw in joined, kw
