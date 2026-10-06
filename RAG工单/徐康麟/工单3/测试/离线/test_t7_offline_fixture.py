# -*- coding: utf-8 -*-
"""离线级：14 题固定输入与不可答负例集的自洽性（对真实 PDF 逐页复核）。

工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化

这是**测试基线**的第一道闸：如果 golden 的 ``evidence_verbatim`` 在它声称的物理页上
找不到，那么后面所有「命中/引用」断言都是在自欺。因此本文件用独立打开的原始 PDF
（``common.pdf_probe``，不经过产品解析链路）反向验证每条 fixtures。

同时锁定两条 captain 复核过的**判据**：
    * PDF2「本次发行募集资金」+「补充流动资金」同页共现 = 空集（N-4 成立的前提）；
    * PDF1 同页共现主锚点 = 物理 479（全库唯一）。
"""

from __future__ import annotations

from typing import Any

import pytest

from common import assertions, golden as golden_mod, negatives as negatives_mod, paths, pdf_probe

pytestmark = [pytest.mark.offline, pytest.mark.linkage]


def _item_by_id(items: list[golden_mod.GoldenItem], question_id: int) -> golden_mod.GoldenItem:
    """按 id 取题（不存在直接失败，避免静默跳过）。"""
    index = golden_mod.by_id(items)
    assert question_id in index, f"golden 缺少题 id={question_id}"
    return index[question_id]


def test_golden_integrity() -> None:
    """14 题齐备、页码合法、语料可自动发现、语料归属正确。"""
    report = golden_mod.integrity_report()
    report.print()
    assert report.ok, "\n".join(c.render() for c in report.failures())


def test_negatives_integrity() -> None:
    """负例集齐备：N-4 编造红线、N-5 互污染三方向、N-6 闸门 fail-open 都在。"""
    report = negatives_mod.fixtures_integrity_report()
    report.print()
    assert report.ok, "\n".join(c.render() for c in report.failures())


def test_evidence_verbatim_found_on_claimed_pages(
    golden_items: list[golden_mod.GoldenItem], discovered_pdfs: list[Any]
) -> None:
    """每条 ``evidence_verbatim`` 必须真的落在其声称的某个物理页上（逐题核验）。"""
    failures: list[str] = []
    for item in golden_items:
        path = golden_mod.corpus_path(item)
        assert path is not None, f"题 {item.id} 的语料无法自动发现"
        ok_pages: list[int] = []
        for page in item.evidence_pages:
            text = pdf_probe.page_text(str(path), page)
            if assertions.support_present(text, item.evidence_verbatim):
                ok_pages.append(page)
        if not ok_pages:
            failures.append(f"题 {item.id}（{path.name}）：evidence_verbatim 在 {item.evidence_pages} 均未命中")
    assert not failures, "\n".join(failures)


def test_citation_quote_found_on_primary_page(
    golden_items: list[golden_mod.GoldenItem],
) -> None:
    """``citation_quote``（引用回溯核验用）必须落在 design 锚点页上。"""
    failures: list[str] = []
    for item in golden_items:
        path = golden_mod.corpus_path(item)
        if path is None or not item.citation_quote:
            continue
        found = [p for p in item.evidence_pages
                 if assertions.support_present(pdf_probe.page_text(str(path), p), item.citation_quote)]
        if not found:
            failures.append(f"题 {item.id}：citation_quote={item.citation_quote!r} 在 {item.evidence_pages} 未命中")
    assert not failures, "\n".join(failures)


def test_question_207_evidence_is_synthetic_and_verbatim_is_real() -> None:
    """题 207：golden ``evidence`` 在真实页找不到（合成串），而 ``evidence_verbatim`` 在 490 命中。"""
    items = golden_mod.load_golden()
    item = _item_by_id(items, 207)
    path = golden_mod.corpus_path(item)
    assert path is not None
    page_490 = pdf_probe.page_text(str(path), 490)
    page_479 = pdf_probe.page_text(str(path), 479)

    assert not assertions.support_present(page_490, item.evidence), \
        "题 207 的 golden evidence 竟然被真实页包含 —— 与 captain 裁定不符，需重新确认基线"
    assert not assertions.support_present(page_479, item.evidence)
    assert assertions.support_present(page_490, item.evidence_verbatim), \
        "题 207 的 evidence_verbatim 未在物理 490 命中"
    assert "15,000" in page_490 and "补充流动资金" in page_490
    assert "补充流动资金" in page_479 and "本次发行募集资金" in page_479


def test_pdf1_cooccurrence_anchor_is_479_only() -> None:
    """PDF1 全文：「本次发行募集资金」+「补充流动资金」同页共现 = 仅物理 479。"""
    path = golden_mod.corpus_path(_item_by_id(golden_mod.load_golden(), 207))
    assert path is not None
    both = [page for page, text in pdf_probe.iter_pages(path)
            if "本次发行募集资金" in text and "补充流动资金" in text]
    assert both == [479], f"同页共现主锚点应为 [479]，实测 {both}"


def test_pdf2_cooccurrence_is_empty_and_340_is_bank_loan() -> None:
    """PDF2 全文同页共现 = 空集（N-4 前提）；物理 340 的「补充流动资金」是银行借款 300 万美元。"""
    path = paths.resolve_corpus("2")
    if path is None:
        pytest.skip("PDF2 语料缺席：N-4 用例自动 pending")
    both = [page for page, text in pdf_probe.iter_pages(path)
            if "本次发行募集资金" in text and "补充流动资金" in text]
    assert both == [], f"PDF2 同页共现应为空集，实测 {both}"
    page_340 = pdf_probe.page_text(str(path), 340)
    assert "补充流动资金" in page_340
    assert "300 万美元" in page_340 and "汉口银行" in page_340


def test_pdf2_question_anchors() -> None:
    """PDF2 四题锚点：22/24 发行股数与比例、22 募投 5 项与占位符、157 两表（赵马克 / 7 家企业）。"""
    path = paths.resolve_corpus("2")
    if path is None:
        pytest.skip("PDF2 语料缺席：id 1~4 自动 pending")
    p22 = pdf_probe.page_text(str(path), 22)
    p24 = pdf_probe.page_text(str(path), 24)
    p157 = pdf_probe.page_text(str(path), 157)
    p21 = pdf_probe.page_text(str(path), 21)

    assert "1,670" in p22 and "25.04%" in p22
    assert "1,670" in p24 and "25.04%" in p24
    for token in ("3,393.40", "1,526.38", "2,492.78", "9,000.00", "其他与主营业务相关的营运资金"):
        assert token in p22, f"PDF2 物理 22 缺少募投要素 {token!r}"
    assert "[ ◆ ]" in p22, "PDF2 物理 22 第 5 项占位符 [ ◆ ] 缺失"

    assert pdf_probe.table_count(str(path), 157) == 2, "物理 157 应为两张表（存在/不存在控制关系）"
    assert "赵马克" in p157 and "42.35%" in p157 and "公司控股股东" in p157
    for company in ("融冰投资", "武汉博润", "上海博润", "听音投资", "联众聚源", "力源贸易", "普芯达"):
        assert company in p157, f"物理 157 缺少企业 {company!r}"
    assert "赵马克" in p21 and "2,117.70" in p21 and "42.35%" in p21
    assert "30,600 万元" not in p157, "不得出现凭空数字（防 fixture 被改坏）"


def test_pdf1_page152_and_153_roles() -> None:
    """题 34/793：「表在 153 ≠ 答案在 152」——152 无表且含答案正文，153 有示意表。"""
    path = paths.resolve_corpus("1")
    assert path is not None
    p152 = pdf_probe.page_text(str(path), 152)
    assert "电子元器件制造企业" in p152 and "金属壳体制造企业" in p152
    assert "军队、政府机关、能源" in p152
    assert pdf_probe.table_count(str(path), 152) == 0, "物理 152 应无表格（正文证据页）"
    assert pdf_probe.table_count(str(path), 153) >= 1, "物理 153 应含上下游示意表（辅助证据页）"


def test_question_texts_are_single_sourced(
    golden_items: list[golden_mod.GoldenItem]
) -> None:
    """题面单一来源：本 fixture 与团队既有 fixture 的**题面必须逐字一致**。

    为什么必须测：题面多写一个字，「14 题准确率」就不再可比 —— T5/T6/T9 的既有数字
    都是基于 ``测试/测试数据/eval_retrieval_14.jsonl`` 的题面跑出来的。
    本用例把「漂移」钉死在测试里（id 1~4 曾出现两版措辞，已统一到 T5 版）。
    """
    import json  # noqa: PLC0415

    peer = paths.TEST_DATA_DIR / "eval_retrieval_14.jsonl"
    if not peer.exists():
        pytest.skip(f"团队既有 fixture 不存在：{paths.display(peer)}")
    peer_rows = {}
    for line in peer.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            peer_rows[int(row["id"])] = row
    mine = {item.id: item for item in golden_items}
    drift: list[str] = []
    for qid, row in peer_rows.items():
        if qid not in mine:
            drift.append(f"id {qid} 只存在于 {peer.name}")
            continue
        if str(row.get("question", "")).strip() != mine[qid].question.strip():
            drift.append(f"id {qid} 题面不一致：\n    既有={row.get('question')!r}\n    本仓={mine[qid].question!r}")
    assert not drift, "\n".join(drift)


def test_dev_golden_kept_in_sync_when_present(golden_items: list[golden_mod.GoldenItem]) -> None:
    """若 T9 的 ``研发/data/eval/golden_qa.jsonl`` 已生成，则题面与页码必须与本 fixture 一致。

    文件不存在时跳过（对应「PDF2 缺席/尚未生成」的合法场景），生成后本用例自动生效，
    防止「评估用一套题、测试用另一套题」。
    """
    import json  # noqa: PLC0415

    dev = paths.EVAL_DIR / "golden_qa.jsonl"
    if not dev.exists():
        pytest.skip(f"研发侧 golden 尚未生成：{paths.display(dev)}")
    dev_rows = {}
    for line in dev.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            dev_rows[int(row["id"])] = row
    mine = {item.id: item for item in golden_items}
    assert set(dev_rows) == set(mine), f"题集不一致：研发 {sorted(dev_rows)} vs 测试 {sorted(mine)}"
    drift: list[str] = []
    for qid, row in dev_rows.items():
        if str(row.get("question", "")).strip() != mine[qid].question.strip():
            drift.append(f"id {qid} 题面不一致：研发={row.get('question')!r} 测试={mine[qid].question!r}")
        dev_pages = sorted(int(p) for p in (row.get("evidence_pages") or []))
        if dev_pages and not set(dev_pages) <= set(mine[qid].evidence_pages):
            drift.append(f"id {qid} 证据页不兼容：研发={dev_pages} 测试={mine[qid].evidence_pages}")
    assert not drift, "\n".join(drift)


def test_page_number_trap_is_documented() -> None:
    """页码陷阱取证：PDF1 页脚 ``1-1-128`` 与 PDF2 页眉第 2 行 ``21`` 都是 0-based 索引。"""
    pdf1 = paths.resolve_corpus("1")
    pdf2 = paths.resolve_corpus("2")
    assert pdf1 is not None
    evidence_129 = pdf_probe.page_number_evidence(str(pdf1), 129)
    assert "1-1-128" in evidence_129["footer_printed"], evidence_129
    assert assertions.looks_like_printed_page_form("1-1-128")
    if pdf2 is not None:
        evidence_22 = pdf_probe.page_number_evidence(str(pdf2), 22)
        assert "21" in evidence_22["header_second_line"], evidence_22
        assert assertions.looks_like_printed_page_form("21")
    assert assertions.printed_page_forms(129) == ["128", "1-1-128"]
