# -*- coding: utf-8 -*-
"""离线级：查询分类口径（N-7 类型分布 / N-8 数值信号与剥离可逆）。

工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化

判定依据（``设计/验收标准.md`` §2.1 N-7 / N-8，v1.5/v1.6 冻结）：
    * 期望主体类型：``any = 10`` / ``person = 2``（题 3、题 531）/ ``organization = 2``（题 4、题 34）；
      必须剥掉发行人全称后再判，**不得**退化成「含『公司』即 organization」；
    * ``expects_numeric``：True = 题 1/3/33/207/260/543；False = 题 2/4/34/95/531/793/795/957；
    * 发行人剥离必须是**纯替换且可逆**（还原后逐字相同），且保留「的多少 / 分别 / 是多少」等数值信号词；
    * 同一题连续 3 次分类结果稳定（保证 T7 多轮测试与 T8 评估可复现）。
"""

from __future__ import annotations

import types
from typing import Any

import pytest

from common import artifacts, assertions, golden as golden_mod

pytestmark = [pytest.mark.offline, pytest.mark.linkage]

#: 契约常量集中在 ``common/assertions.py``（单一实现，避免各测试各写一份）
EXPECTED_SUBJECT_TYPE = assertions.EXPECTED_SUBJECT_TYPE
NUMERIC_TRUE = set(assertions.NUMERIC_TRUE_IDS)
NUMERIC_FALSE = set(assertions.NUMERIC_FALSE_IDS)
PREFER_TABLE = set(assertions.PREFER_TABLE_IDS)
KEEP_TOKENS = assertions.KEEP_SIGNAL_TOKENS


@pytest.fixture(scope="session")
def issuer_names(app_config: Any) -> list[str]:
    """自动发现的发行人全称（来自两份 PDF 首页，禁硬编码）。"""
    from app.core.config import discover_issuer_names  # noqa: PLC0415

    names = list(discover_issuer_names())
    assert names, "未能发现任何发行人全称（discover_issuer_names 返回空）"
    return names


def test_subject_expected_type_distribution(golden_items: list[Any], issuer_names: list[str]) -> None:
    """N-7：14 题期望类型分布必须是 any 10 / person 2 / organization 2。"""
    from app.core.answerability import classify_subject_expectation  # noqa: PLC0415

    actual = {item.id: classify_subject_expectation(item.question, issuer_names=issuer_names)
              for item in golden_items}
    mismatch = {key: (value, actual.get(key)) for key, value in EXPECTED_SUBJECT_TYPE.items()
                if actual.get(key) != value}
    counts: dict[str, int] = {}
    for value in actual.values():
        counts[value] = counts.get(value, 0) + 1
    assert not mismatch, f"类型判定不符（期望/实测）：{mismatch}"
    assert counts == {"any": 10, "person": 2, "organization": 2}, f"分布应为 any10/person2/org2，实测 {counts}"


def test_expects_numeric_signal(golden_items: list[Any]) -> None:
    """N-8：数值信号集合必须精确匹配（分类取自**原题**）。"""
    from app.core.query_understanding import classify_question  # noqa: PLC0415

    wrong: list[str] = []
    for item in golden_items:
        field_type, expects_numeric, prefer_table = classify_question(item.question)
        expected = True if item.id in NUMERIC_TRUE else (False if item.id in NUMERIC_FALSE else None)
        if expected is not None and bool(expects_numeric) is not expected:
            wrong.append(f"题 {item.id} expects_numeric 期望 {expected}，实测 {expects_numeric}（field={field_type}）")
    assert not wrong, "\n".join(wrong)

    prefer = {item.id for item in golden_items if classify_question(item.question)[2]}
    assert PREFER_TABLE <= prefer, f"prefer_table_retrieval 应包含 {sorted(PREFER_TABLE)}，实测 {sorted(prefer)}"


def test_strip_issuer_names_reversible(golden_items: list[Any], issuer_names: list[str]) -> None:
    """剥离契约：纯替换且可逆（还原后与原文逐字相同），且保留数值信号词。"""
    from app.core.answerability import strip_issuer_names_checked  # noqa: PLC0415

    problems: list[str] = []
    for item in golden_items:
        stripped, ok = strip_issuer_names_checked(item.question, issuer_names)
        if not ok:
            problems.append(f"题 {item.id} 可逆校验失败：{stripped!r}")
            continue
        restored = [stripped.replace("该公司", name) for name in issuer_names]
        if item.question not in restored:
            problems.append(f"题 {item.id} 还原后与原文不一致：{stripped!r}")
        for token in KEEP_TOKENS.get(item.id, ()):  # 数值信号词不得丢失
            if token not in stripped:
                problems.append(f"题 {item.id} 剥离后丢失信号词 {token!r}：{stripped!r}")
    assert not problems, "\n".join(problems)


def test_classification_is_stable_across_repeats(golden_items: list[Any], issuer_names: list[str]) -> None:
    """可复现性：同一题连续 3 次分类结果完全一致（field_type / expects_numeric / prefer_table）。"""
    from app.core.answerability import classify_subject_expectation  # noqa: PLC0415
    from app.core.query_understanding import classify_question  # noqa: PLC0415

    unstable: list[str] = []
    for item in golden_items:
        first = classify_question(item.question)
        subject = classify_subject_expectation(item.question, issuer_names=issuer_names)
        for _ in range(2):
            if classify_question(item.question) != first or \
                    classify_subject_expectation(item.question, issuer_names=issuer_names) != subject:
                unstable.append(f"题 {item.id} 分类结果不稳定")
                break
    assert not unstable, "\n".join(unstable)


def test_multiturn_followup_numeric_signal(issuer_names: list[str]) -> None:
    """N-8 多轮：轮 2「那法定代表人呢？」的数值信号必须取自本轮原文（False）。"""
    from app.core.query_understanding import classify_question  # noqa: PLC0415

    turn1 = "武汉兴图新科电子股份有限公司注册资本是多少？"
    turn2 = "那法定代表人呢？"
    assert bool(classify_question(turn1)[1]) is True
    assert bool(classify_question(turn2)[1]) is False
    assert isinstance(issuer_names, list)


# ---------------------------------------------------------------------------
# 防漂移第二道闸（captain 裁定 1）：题面 + 预期分类 + 预期 allowed 集合
# ---------------------------------------------------------------------------
def test_question_contract_matches_golden(golden_items: list[Any]) -> None:
    """题 3/题 4 的**题面**必须与 ``assertions.QUESTION_CONTRACT`` 逐字一致。

    题面多一个字，「同页互污染」的实验条件就变了（T6 期间题面曾被改过，
    导致题 4 在权威产物里被闸门误判 ``org_leaked``）。此闸把题面钉死。
    """
    index = golden_mod.by_id(golden_items)
    problems: list[str] = []
    for question_id, contract in assertions.QUESTION_CONTRACT.items():
        item = index.get(question_id)
        if item is None:
            problems.append(f"golden 缺少题 {question_id}")
            continue
        check = assertions.check_question_text(question_id, item.question)
        if not check.ok:
            problems.append(check.render())
        # 语料与证据页也要对齐契约（本 fixture 的证据页为契约页的超集）
        if item.corpus != contract["corpus"]:
            problems.append(f"题 {question_id} 语料应为 {contract['corpus']}，实测 {item.corpus}")
        if not set(contract["evidence_pages"]) <= set(item.evidence_pages):
            problems.append(f"题 {question_id} 证据页缺 {sorted(set(contract['evidence_pages']) - set(item.evidence_pages))}")
    assert not problems, "\n".join(problems)


def test_expected_subject_type_is_callable_contract() -> None:
    """契约常量可被用例直接查询（题 3 = person、题 4 = organization、allowed 必备集合）。"""
    assert assertions.expected_subject_type(3) == "person"
    assert assertions.expected_subject_type(4) == "organization"
    assert assertions.expected_allowed_subset(3) == ("赵马克",)
    assert assertions.expected_allowed_subset(4) == assertions.SEVEN_COMPANIES
    assert assertions.FALSE_REFUSAL_FIRST_CASE == 4


def test_gate_allowed_sets_from_evidence_tables() -> None:
    """题 3/题 4 的 allowed 集合必须能从**物理 157 的两张真实表**推出来。

    这是闸门判定的地基：如果 allowed 集合本身是错的（例如把自然人算进企业桶），
    题 4 必然被误拒答（``subject_gate:org_leaked``）。因此这里对**表块**做确定性断言，
    与 LLM 无关。
    """
    from app.core.answerability import allowed_set_from_tables  # noqa: PLC0415

    blocks = artifacts.tables_on_page("pdf2", 157)
    assert blocks, "物理 157 未解析出表格块（关联方两表）"
    person_md = [str(b.get("markdown") or "") for b in blocks if "赵马克" in str(b.get("markdown") or "")]
    org_md = [str(b.get("markdown") or "") for b in blocks if "融冰投资" in str(b.get("markdown") or "")]
    assert person_md, "物理 157 未找到「存在控制关系的关联方」表"
    assert org_md, "物理 157 未找到「不存在控制关系的关联方」表"

    person_allowed, _person_other = allowed_set_from_tables(
        [types.SimpleNamespace(markdown=md, content=md) for md in person_md], "person")
    org_allowed, _org_other = allowed_set_from_tables(
        [types.SimpleNamespace(markdown=md, content=md) for md in org_md], "organization")

    problems: list[str] = []
    if "赵马克" not in person_allowed:
        problems.append(f"题 3（person）allowed 缺自然人赵马克：{person_allowed}")
    if any(company in person_allowed for company in assertions.SEVEN_COMPANIES):
        problems.append(f"题 3（person）allowed 混入企业：{[c for c in assertions.SEVEN_COMPANIES if c in person_allowed]}")
    missing = [c for c in assertions.SEVEN_COMPANIES if c not in org_allowed]
    if missing:
        problems.append(f"题 4（organization）allowed 缺企业：{missing}（实测 {org_allowed}）")
    if "赵马克" in org_allowed:
        problems.append("题 4（organization）allowed 混入自然人赵马克（会导致闸门误判）")
    assert not problems, "\n".join(problems)
