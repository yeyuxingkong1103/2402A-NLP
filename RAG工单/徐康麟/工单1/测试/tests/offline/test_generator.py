"""离线测试：答案生成与引用（``app/core/generator.py`` / ``app/core/citation.py``）。

测试目标（工单 9.1 / 5.5 / 5.6 / 5.8）：
1. **有证据时必须给出基于原文的答案**（抽取式路径，不依赖任何 LLM 服务）；
2. **无证据时必须回复"不清楚"**，且不得编造、不得带引用；
3. 引用格式 ``[页码: N]`` 正确，页码来自证据片段，越界引用能被识别（幻觉引用）；
4. 抽取式答案覆盖 10 道工单问题涉及的 8 类意图规则；
5. ``render_markdown`` 能输出带引用清单的 Markdown。

本文件全部使用**手工构造的检索片段**（内容取自 ``golden_qa.jsonl`` 的原文依据），
因此不受索引是否已生成的影响，也不依赖 LLM。
"""

from __future__ import annotations

import re

import pytest

from app.core.citation import CitationManager
from app.core.config import get_settings
from app.core.generator import Generator
from app.models.schemas import Answer, Chunk, Citation, RetrievedChunk

CITATION_LABEL_PATTERN = re.compile(r"^\[页码: \d+\]$")

# --------------------------------------------------------------------------
# 取自招股说明书的原文依据（与 data/eval/golden_qa.jsonl 的 evidence 一致）
# --------------------------------------------------------------------------
EVIDENCE_BASIC_INFO = (
    "（一）发行人基本情况\n中文名称\n武汉兴图新科电子股份有限公司\n"
    "法定代表人\n程家明\n注册资本\n5,520.00 万元\n"
    "注册地址\n湖北省武汉市东湖新技术开发区关山大道1 号软件产业三期A3 栋8 层"
)
EVIDENCE_REVENUE = (
    "发行人产品主要应用于军队的信息化建设，报告期内，公司来自军用领域的收入分别为"
    "6,464.51 万元、14,414.16 万元、18,780.67 万元和4,627.14 万元，"
    "占主营业务收入比重分别为82.10%、97.31%、94.84%和94.34%。"
)
EVIDENCE_INDUSTRY = (
    "1、行业上下游情况\n电子信息行业的上游涉及信息系统相关的电子元器件制造企业，"
    "以及机箱、机柜等金属壳体制造企业，竞争充分，采购便利。"
    "下游行业为各类终端用户，覆盖范围广泛，主要包括军队、政府机关、能源等行业企业。"
)
EVIDENCE_AWARD = (
    "2014年12月，某大型研究所牵头承担的“某情报、指挥、控制与通信网络一体化工程”"
    "（即相当于美军的C4ISR系统）荣获国家科技进步一等奖。"
)
EVIDENCE_STANDARD = (
    "公司目前已经成为军队视频指挥领域的重要供应商，参与制定了全军第一个视频指挥系统技术标准"
    "（即2019年制订的《某视频指挥系统技术规范（1.0版）》）。"
)
EVIDENCE_FUND_TEXT = (
    "本次募集资金投资项目：基于云联邦架构的军用视频指挥平台升级及产业化项目、"
    "研发中心建设项目、补充流动资金15,000.00万元。"
)
EVIDENCE_FUND_TABLE = (
    "[表格 p479_t1] 第 479 页\n"
    "| 项目名称 | 投资金额 |\n| --- | --- |\n| 补充流动资金 | 15,000.00 |"
)


def _context(page: int, content: str, score: float = 1.0, chunk_type: str = "text", chunk_id: str = "") -> RetrievedChunk:
    """构造一个检索片段（离线用例用，避免依赖索引）。"""
    chunk = Chunk(
        chunk_id=chunk_id or f"c{page:06d}",
        doc_id="doc_test",
        page=page,
        section="测试章节",
        type=chunk_type,  # type: ignore[arg-type]
        content=content,
        char_count=len(content),
    )
    return RetrievedChunk(chunk=chunk, score=score, vector_score=score, bm25_score=0.0, source="hybrid")


@pytest.fixture
def generator() -> Generator:
    """强制抽取式的生成器：离线可跑，且结果完全可复现。"""
    return Generator(force_extractive=True)


# ==========================================================================
# 1. 无证据 -> 不清楚
# ==========================================================================
def test_no_context_returns_unknown(generator) -> None:
    """没有检索片段时必须回复统一的"不清楚"，且不得带引用。"""
    answer = generator.generate("武汉兴图新科电子股份有限公司注册资本是多少？", contexts=[])
    assert answer.is_unknown is True, "无证据时必须走兜底回复"
    assert answer.answer == get_settings().app.unknown_answer, (
        f"兜底文案应为 {get_settings().app.unknown_answer!r}，实际为 {answer.answer!r}"
    )
    assert answer.citations == [], "兜底回复不应携带任何引用"
    assert answer.unknown_reason, "兜底回复必须记录原因（禁止静默失败）"


def test_unknown_is_not_fabricated_when_evidence_irrelevant(generator) -> None:
    """证据与问题无关时，抽取式回答要么命中规则、要么给出有出处的摘要，不得凭空编造数字。"""
    irrelevant = [_context(300, "公司的办公地址位于武汉市东湖新技术开发区，办公场所为租赁取得。")]
    answer = generator.generate("武汉兴图新科电子股份有限公司注册资本是多少？", contexts=irrelevant)
    assert isinstance(answer.answer, str) and answer.answer.strip(), "生成结果不得为空"
    assert "5,520" not in answer.answer, "证据中不含注册资本，答案却编造出了注册资本数字"


# ==========================================================================
# 2. 有证据 -> 命中受控规则
# ==========================================================================
def test_extractive_registered_capital(generator) -> None:
    """注册资本意图：应从证据中抽取出"5,520 万元"。"""
    contexts = [_context(22, EVIDENCE_BASIC_INFO)]
    answer = generator.generate("武汉兴图新科电子股份有限公司注册资本是多少？", contexts=contexts)
    assert answer.is_unknown is False, "证据中含注册资本，不应回复不清楚"
    assert "5,520" in answer.answer, f"答案未包含注册资本 5,520 万元，实际为：{answer.answer!r}"
    assert answer.mode == "extractive", f"离线模式应走抽取式，实际 mode={answer.mode}"
    assert answer.citations and answer.citations[0].page == 22, "引用页码应指向证据所在页 22"


def test_extractive_legal_representative(generator) -> None:
    """法定代表人意图：应抽取到"程家明"。"""
    contexts = [_context(22, EVIDENCE_BASIC_INFO)]
    answer = generator.generate("武汉兴图新科电子股份有限公司法定代表人是谁？", contexts=contexts)
    assert answer.is_unknown is False, "证据中含法定代表人信息，不应回复不清楚"
    assert "程家明" in answer.answer, f"答案未包含法定代表人姓名，实际为：{answer.answer!r}"


def test_extractive_military_revenue(generator) -> None:
    """军用领域收入意图：四期金额必须完整抽取。"""
    contexts = [_context(129, EVIDENCE_REVENUE)]
    answer = generator.generate("报告期内，公司来自军用领域的收入分别是多少？", contexts=contexts)
    assert answer.is_unknown is False, "证据中含军用领域收入，不应回复不清楚"
    for amount in ("6,464.51", "14,414.16", "18,780.67", "4,627.14"):
        assert amount in answer.answer, f"答案缺少收入金额 {amount} 万元，实际为：{answer.answer!r}"


def test_extractive_revenue_ratio(generator) -> None:
    """收入占比意图：四期比重必须完整抽取。"""
    contexts = [_context(129, EVIDENCE_REVENUE)]
    answer = generator.generate("报告期内，公司来自军用领域的收入占主营业务收入的比重分别是多少？", contexts=contexts)
    assert answer.is_unknown is False, "证据中含收入占比，不应回复不清楚"
    for percent in ("82.10%", "97.31%", "94.84%", "94.34%"):
        assert percent in answer.answer, f"答案缺少比重 {percent}，实际为：{answer.answer!r}"


def test_extractive_upstream_and_downstream(generator) -> None:
    """上下游意图：上游企业类型与下游行业都要能抽取。"""
    contexts = [_context(152, EVIDENCE_INDUSTRY)]

    upstream = generator.generate("招股意向书中，电子信息行业的上游涉及哪些企业？", contexts=contexts)
    assert "电子元器件制造企业" in upstream.answer, f"上游答案不完整：{upstream.answer!r}"

    downstream = generator.generate("招股意向书中，电子信息行业的下游主要包括哪些行业？", contexts=contexts)
    assert "军队" in downstream.answer and "政府机关" in downstream.answer, (
        f"下游答案不完整：{downstream.answer!r}"
    )


def test_extractive_supplier_field_and_standard(generator) -> None:
    """行业地位与技术标准意图：领域名称与技术标准名称都要能抽取。"""
    contexts = [_context(160, EVIDENCE_STANDARD)]

    supplier = generator.generate("公司在哪个领域已经成为重要供应商？", contexts=contexts)
    assert "重要供应商" in supplier.answer, f"未抽取到领域信息：{supplier.answer!r}"
    assert "视频指挥" in supplier.answer, f"领域名称不完整：{supplier.answer!r}"

    standard = generator.generate("公司参与制定了哪个技术标准？", contexts=contexts)
    assert "视频指挥系统技术标准" in standard.answer, f"技术标准答案不完整：{standard.answer!r}"


def test_extractive_award(generator) -> None:
    """荣誉奖项意图：应抽取到荣获国家科技进步一等奖的工程名称。"""
    contexts = [_context(157, EVIDENCE_AWARD)]
    answer = generator.generate("公司参与的哪个工程荣获了国家科技进步一等奖？", contexts=contexts)
    assert "荣获国家科技进步一等奖" in answer.answer, f"答案未包含奖项信息：{answer.answer!r}"
    assert "一体化工程" in answer.answer, f"答案未包含工程名称：{answer.answer!r}"


def test_extractive_fund_usage_from_text(generator) -> None:
    """募资用途意图（正文形式）：应抽取到补充流动资金的金额。"""
    contexts = [_context(479, EVIDENCE_FUND_TEXT)]
    answer = generator.generate("公司计划使用本次发行募集资金的多少用于补充流动资金？", contexts=contexts)
    assert "15,000.00" in answer.answer, f"未抽取到补充流动资金金额：{answer.answer!r}"


def test_extractive_fund_usage_from_table(generator) -> None:
    """募资用途意图（表格形式）：整表 chunk 中的金额也要能抽取。"""
    contexts = [_context(479, EVIDENCE_FUND_TABLE, chunk_type="table")]
    answer = generator.generate("公司计划使用本次发行募集资金的多少用于补充流动资金？", contexts=contexts)
    assert "15,000.00" in answer.answer, f"未从表格中抽取到金额：{answer.answer!r}"
    assert answer.citations and answer.citations[0].page == 479, "表格证据的引用页码应为 479"


@pytest.mark.xfail(
    reason=(
        "已知缺陷：_answer_fund_usage 取表格行中第一个'纯数字'单元格作为金额"
        "（app/core/generator.py:389-400），招股书募集资金表首列常为『序号』，"
        "会把序号当成金额（实测返回『3 万元』而非『15,000.00 万元』）"
    ),
    strict=False,
)
def test_extractive_fund_usage_table_with_index_column(generator) -> None:
    """表格首列是「序号」时，必须抽取真正的金额列，而不是序号。"""
    table = (
        "[表格 p479_t1] 第 479 页\n"
        "| 序号 | 项目名称 | 拟投入募集资金 |\n| --- | --- | --- |\n"
        "| 3 | 补充流动资金 | 15,000.00 |"
    )
    contexts = [_context(479, table, chunk_type="table")]
    answer = generator.generate("公司计划使用本次发行募集资金的多少用于补充流动资金？", contexts=contexts)
    assert "15,000.00" in answer.answer, (
        f"未抽到真实金额（应含 15,000.00），实际答案：{answer.answer!r}"
    )


# ==========================================================================
# 3. 引用格式与校验
# ==========================================================================
def test_answer_carries_valid_citations(generator) -> None:
    """生成答案的引用必须格式正确、页码来自证据、且通过范围校验。"""
    contexts = [_context(129, EVIDENCE_REVENUE), _context(22, EVIDENCE_BASIC_INFO)]
    answer = generator.generate("报告期内，公司来自军用领域的收入分别是多少？", contexts=contexts)

    assert answer.citations, "有答案的回复必须带引用，保证可追溯"
    pages = {item.chunk.page for item in contexts}
    for citation in answer.citations:
        assert CITATION_LABEL_PATTERN.match(citation.label()), f"引用标签格式错误：{citation.label()!r}"
        assert citation.page in pages, f"引用页码 {citation.page} 不在证据页 {sorted(pages)} 中"
        assert citation.chunk_id, "引用必须携带 chunk_id 以便前端展开原文"
        assert citation.snippet, "引用必须携带原文片段摘要"

    manager = CitationManager(range(1, 549))
    validation = manager.validate(answer)
    assert validation["total"] == len(answer.citations), "校验统计的引用总数与答案不一致"
    assert validation["valid"] == validation["total"], f"存在越界引用：{validation['invalid_pages']}"
    assert validation["valid_ratio"] == 1.0, f"引用合法率应为 1.0，实际 {validation['valid_ratio']}"


def test_citation_label_and_page_extraction() -> None:
    """``[页码: N]`` 标签与页码提取必须稳定（含空格差异与去重）。"""
    assert Citation(page=12, chunk_id="c1", snippet="x").label() == "[页码: 12]", "引用标签格式错误"
    assert CitationManager.extract_pages("[页码: 12]……[页码:12]，另见 [页码: 130]") == [12, 130], (
        "页码提取应支持空格差异、去重并保持出现顺序"
    )
    assert CitationManager.extract_pages("本回答没有任何引用") == [], "无引用时应返回空列表"


def test_citation_build_prefers_explicit_pages() -> None:
    """答案正文里显式写了 [页码: N] 时，该页必须排在最前（其余片段只能作为补充）。"""
    contexts = [_context(129, EVIDENCE_REVENUE, score=0.9), _context(22, EVIDENCE_BASIC_INFO, score=0.5)]
    manager = CitationManager(range(1, 549))
    citations = manager.build("公司注册资本为 5,520 万元 [页码: 22]", contexts)

    assert citations, "显式引用了页码却没有任何引用输出"
    assert citations[0].page == 22, (
        f"显式引用的第 22 页应排在首位，实际顺序为 {[c.page for c in citations]}"
    )
    assert citations[0].chunk_id == "c000022", "引用应指向第 22 页的片段"
    assert {c.page for c in citations} <= {22, 129}, (
        f"引用页码只能来自候选片段，实际为 {[c.page for c in citations]}"
    )

    limited = manager.build("公司注册资本为 5,520 万元 [页码: 22]", contexts, limit=1)
    assert [c.page for c in limited] == [22], f"limit=1 时应只保留显式引用的页码，实际 {[c.page for c in limited]}"


def test_citation_build_falls_back_to_top_contexts() -> None:
    """答案中没有显式引用时，回退为按检索分数取前若干条证据。"""
    contexts = [_context(129, EVIDENCE_REVENUE, score=0.9), _context(22, EVIDENCE_BASIC_INFO, score=0.5)]
    manager = CitationManager(range(1, 549))
    citations = manager.build("报告期内军用领域收入见原文。", contexts)
    assert {citation.page for citation in citations} == {129, 22}, (
        f"回退引用应覆盖全部候选页，实际为 {[c.page for c in citations]}"
    )
    scores = [citation.score for citation in citations]
    assert scores == sorted(scores, reverse=True), f"回退引用应按分数降序，实际为 {scores}"


def test_citation_never_fabricates_chunk_ids() -> None:
    """模型引用了候选中不存在的页码（幻觉引用）时，绝不能凭空造出一条带 chunk_id 的引用。"""
    contexts = [_context(129, EVIDENCE_REVENUE)]
    manager = CitationManager(range(1, 549))
    citations = manager.build("该数据见招股书 [页码: 999]", contexts)

    hallucinated = [citation for citation in citations if citation.page == 999]
    assert not hallucinated or not hallucinated[0].chunk_id, (
        "第 999 页不在候选片段中，不能为它生成带 chunk_id 的引用（属于编造引用）"
    )
    for citation in citations:
        assert citation.chunk_id, "所有引用都必须能追溯到真实片段"
        assert citation.page in {item.chunk.page for item in contexts}, (
            f"引用页码 {citation.page} 不在候选页 {sorted({item.chunk.page for item in contexts})} 中"
        )


def test_citation_validation_flags_out_of_range_pages() -> None:
    """越界页码必须能被 validate 统计出来（评估『引用正确率』依赖该能力）。"""
    manager = CitationManager(range(1, 549))
    answer = Answer(answer="该数据见招股书 [页码: 999]", citations=[Citation(page=999, chunk_id="", snippet="")])
    validation = manager.validate(answer)
    assert validation["invalid_pages"] == [999], f"越界页码未被识别：{validation}"
    assert validation["valid_ratio"] == 0.0, "全部引用越界时合法率应为 0"


def test_render_markdown_contains_citation_list(generator) -> None:
    """Markdown 渲染必须包含答案正文、引用清单与 chunk_id。"""
    contexts = [_context(129, EVIDENCE_REVENUE)]
    answer = generator.generate("报告期内，公司来自军用领域的收入分别是多少？", contexts=contexts)
    markdown = CitationManager(range(1, 549)).render_markdown(answer)

    assert answer.answer in markdown, "渲染结果缺少答案正文"
    assert "**引用来源**" in markdown, "渲染结果缺少引用来源小节"
    assert "[页码: 129]" in markdown, "渲染结果缺少 [页码: N] 标签"
    assert "c000129" in markdown, "渲染结果缺少 chunk_id，前端无法展开原文"


def test_render_markdown_marks_unknown_answer() -> None:
    """兜底回答渲染时必须显式标注"无引用"，避免看起来像有出处的答案。"""
    answer = Answer(answer="不清楚", citations=[], is_unknown=True)
    markdown = CitationManager(range(1, 549)).render_markdown(answer)
    assert "不清楚" in markdown, "渲染结果缺少兜底文案"
    assert "无引用" in markdown, "兜底回答应显式说明没有引用来源"
