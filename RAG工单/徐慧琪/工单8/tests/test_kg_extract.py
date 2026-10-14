# 工单编号：人工智能NLP-RAG-基于GraphRAG实现金融问答
from src.kg_extract import (extract_block, extract_periods, parse_amount)

CHUNK = {
    "chunk_id": "c000123",
    "text": "平安银行2019年末不良贷款率为1.65%，较上年末下降0.10个百分点。"
            "本行2019年实现营业收入1,379.58亿元，同比增长18.2%。",
    "source": "2020-02-14__平安银行__000001__2019年度报告.pdf",
    "page_idx": 23,
    "heading_path": ["主要会计数据"],
    "block_type": "text",
}


def test_parse_amount_with_unit():
    got = parse_amount("营业收入1,379.58亿元", 4)
    assert got is not None
    assert abs(got["value"] - 1379.58) < 1e-6
    assert got["unit"] == "亿元"


def test_parse_amount_plain_number_returns_none_without_unit():
    """没有单位、也不在指标上下文中的裸数字不应被当作金额。"""
    assert parse_amount("第 1379 页", 0) is None


def test_extract_periods():
    periods = {p["period"] for p in extract_periods("2019年末不良贷款率为1.65%")}
    assert "2019年末" in periods


def test_extract_block_finds_company_metric_and_value():
    got = extract_block(CHUNK)
    ids = {e.entity_id for e in got["entities"]}
    assert "Company:平安银行" in ids
    assert any(i.startswith("Metric:不良贷款率") for i in ids), ids

    metric = next(e for e in got["entities"] if e.type == "Metric")
    assert metric.attrs["value"] == 1.65
    assert metric.attrs["unit"] == "%"
    assert metric.attrs["source_doc"] == CHUNK["source"]
    assert metric.attrs["page"] == 23


def test_extract_block_relations_carry_provenance():
    got = extract_block(CHUNK)
    rels = [r for r in got["relations"] if r.rel_type == "HAS_METRIC"]
    assert rels, "应抽出 HAS_METRIC 关系"
    r = rels[0]
    assert r.source_doc == CHUNK["source"]
    assert r.page == 23
    assert r.evidence, "规则抽取的边也必须带原文片段"
    assert r.confidence == 1.0


def test_extract_block_ignores_chunk_without_entities():
    got = extract_block({"chunk_id": "c9", "text": "目录", "source": "x.pdf",
                         "page_idx": 0, "heading_path": [], "block_type": "text"})
    assert got["entities"] == []
    assert got["relations"] == []


def test_metric_entity_id_is_set_specific_and_deduped_in_block():
    """同一指标在同一块中出现多次，只应产生一个 Metric 实体。"""
    chunk = dict(CHUNK, text="不良贷款率1.65%，不良贷款率较上年下降。")
    got = extract_block(chunk)
    metrics = [e for e in got["entities"] if e.type == "Metric"]
    assert len(metrics) == 1


def test_company_metric_uses_nearest_preceding_company():
    chunk = dict(CHUNK, text="招商银行不良贷款率1.16%；平安银行不良贷款率1.65%。")
    got = extract_block(chunk)
    pairs = {(r.head_id, r.attrs.get("value")) for r in got["relations"]
             if r.rel_type == "HAS_METRIC"}
    assert ("Company:招商银行", 1.16) in pairs
    assert ("Company:平安银行", 1.65) in pairs


def test_sub_metric_by_product_is_rejected():
    """「企业贷款不良率」是子类口径，不得记到总口径「不良贷款率」名下。

    否则 6.07% 这类子行业/子品种取值会冒充公司总口径指标，污染问答答案。
    """
    chunk = dict(CHUNK, text="平安银行企业贷款不良率2.13%，较上年末上升0.30个百分点。")
    got = extract_block(chunk)
    assert not [e for e in got["entities"] if e.type == "Metric"]


def test_sub_metric_by_industry_is_rejected():
    chunk = dict(CHUNK, text="招商银行制造业不良贷款率6.07%，房地产业不良贷款率1.29%。")
    got = extract_block(chunk)
    assert not [e for e in got["entities"] if e.type == "Metric"]


def test_total_metric_kept_when_sub_metric_in_same_block():
    """总口径与子类口径同块出现时，只保留总口径的取值。"""
    chunk = dict(CHUNK, text="平安银行不良贷款率1.65%，其中企业贷款不良率2.13%。")
    got = extract_block(chunk)
    metrics = [e for e in got["entities"] if e.type == "Metric"]
    assert len(metrics) == 1
    assert metrics[0].attrs["value"] == 1.65


def test_qualifier_outside_lookbehind_does_not_block():
    """限定词距指标名超过回看窗口、且不在同一子句时，按总口径正常抽取。"""
    chunk = dict(CHUNK,
                 text="本行持续加大企业贷款投放力度，资产质量保持稳定。"
                      "不良贷款率为1.65%。")
    got = extract_block(chunk)
    metrics = [e for e in got["entities"] if e.type == "Metric"]
    assert [m.attrs["value"] for m in metrics] == [1.65]


def test_sub_metric_in_clause_with_loan_phrase_is_rejected():
    """子句内先出现「限定词+贷款/余额」子组合时，后续裸指标名同属该口径。

    语料实例（招商银行 2019 年报 P54）：
    「零售不良贷款余额170.54亿元，不良贷款率0.73%」——0.73% 是零售口径，
    不是公司总口径；同句的「小微不良贷款……不良贷款率0.81%」同理。
    """
    chunk = dict(CHUNK, text=(
        "截至报告期末，本公司零售不良贷款余额170.54亿元，"
        "不良贷款率0.73%，较上年末下降0.06个百分点，其中，小微不良"
        "贷款处置加快，不良贷款率0.81%，较上年末下降0.53个百分点。"))
    got = extract_block(chunk)
    assert not [e for e in got["entities"] if e.type == "Metric"]


def test_sub_metric_enterprise_size_is_rejected():
    """「境内行标大型企业贷款余额…不良贷款率1.91%」是分客户规模口径。

    语料实例（招商银行 2019 年报 P56）；中型企业贷款 5.16% 同理。
    """
    chunk = dict(CHUNK, text=(
        "截至报告期末，本公司境内行标大型企业贷款余额12,697.37亿元，较年初增长"
        "5.92%，占境内公司贷款的85.47%，较年初下降0.45个百分点，不良贷款率"
        "1.91%，较年初下降0.07个百分点；境内行标中型企业贷款余额1,209亿元，"
        "不良贷款率5.16%。"))
    got = extract_block(chunk)
    assert not [e for e in got["entities"] if e.type == "Metric"]


def test_sub_metric_caught_via_parent_text_when_chunk_splits_sentence():
    """分块把「限定词+贷款余额」切在上一块时，用 parent_text 还原子句上下文。

    语料实例（招商银行 2019 年报 P56，本块从「较年初下降0.86个百分点」起）：
    中型企业贷款余额…不良贷款率5.16% —— 限定词在 parent_text 中。
    """
    parent = ("百分点；境内行标中型企业贷款余额1,209亿元，较年初下降3.74%，"
              "占境内公司贷款的8.14%，较年\n初下降0.86个百分点，不良贷款率"
              "5.16%，较年初下降1.28个百分点；境内行标小企业贷款")
    chunk = dict(CHUNK, text=parent[parent.index("较年\n初下降0.86"):],
                 parent_text=parent)
    got = extract_block(chunk)
    assert not [e for e in got["entities"] if e.type == "Metric"]


def test_post_parenthetical_qualifier_is_rejected():
    """指标名紧跟括号限定（不良贷款率（包括商业票据）0.51%）是子类口径。

    语料实例（招商银行 2019 年报 P65，招商永隆集团口径）。
    """
    chunk = dict(CHUNK, text=("本公司控股的招商永隆集团客户存款港币2,431.36亿元，"
                              "贷存比率70.20%，不良贷款率（包括商 业票据）0.51%。"))
    got = extract_block(chunk)
    assert not [e for e in got["entities"] if e.type == "Metric"]


def test_total_metric_in_later_clause_is_kept():
    """子类口径在上一子句，本子句的总口径指标必须保留（不误伤）。"""
    chunk = dict(CHUNK, text=(
        "本公司零售不良贷款余额170.54亿元，不良贷款率0.73%。"
        "报告期末，本集团不良贷款率为1.16%，较上年末下降0.20个百分点。"))
    got = extract_block(chunk)
    metrics = [e for e in got["entities"] if e.type == "Metric"]
    assert [m.attrs["value"] for m in metrics] == [1.16]


def test_table_cross_row_sub_metric_is_rejected():
    """表格跨行子类口径：「个人贷款不良贷款余额…| 不良贷款率 | 0.99%」。

    语料实例（邮储银行 2019 年报 P82 资产质量分品种表）。
    """
    chunk = dict(CHUNK, block_type="table", text=(
        "| 个人贷款不良贷款余额 | | --- | --- |\n"
        "| 273.32 亿元，较上年末增加 25.23 亿元， | 不良贷款率 | 0.99% ，"
        "较上年末下降 0.05 |"))
    got = extract_block(chunk)
    assert not [e for e in got["entities"] if e.type == "Metric"]


def test_table_chunk_keeps_total_metric_without_clause_rule():
    """表格块不做子句级判定：单元格里出现限定词不得误伤总口径取值。"""
    chunk = dict(CHUNK, block_type="table", text=(
        "| 资产质量指标 | 2019年 | 2018年 | 2017年 |\n"
        "| 零售贷款占比 | 52.61% | 51.09% | 49.73% |\n"
        "| 不良贷款率 | 1.65% | 1.75% | 1.70% |"))
    got = extract_block(chunk)
    values = sorted(m.attrs["value"] for m in got["entities"] if m.type == "Metric")
    assert 1.65 in values
