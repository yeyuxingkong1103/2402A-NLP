"""T3 离线测试 ①：PDF 解析层。

工单：人工智能NLP-RAG-基于PDF文档的问答系统优化
覆盖（设计/验收标准.md 验收 11、工单 6.1）：
- 页数、表格数、结构化块数真实；
- 表格块含 Markdown 表结构，``page`` / ``table_id`` 存在且页码在文档范围内；
- 结构化条目字段齐全（``page`` / ``section`` / ``type`` / ``content``）；
- 跨页表格合并生效；
- 能定位到 golden 证据所在页（证据原文可落到具体块）。

真实数据来源：``研发/data/processed/招股说明书1.parsed.json``（548 页，工单1 实测同源）。
"""

from __future__ import annotations

import re

from conftest import norm_b

#: 表格 ID 形如 ``p152_t1``
TABLE_ID_RE = re.compile(r"^p(\d{1,4})_t\d+$")
#: Markdown 表格分隔行
MD_TABLE_RE = re.compile(r"\|\s*---")


def test_page_count_matches_pdf(parsed_document):
    """页数必须与 PDF 真实页数一致（引用页码合法性的前提）。"""
    assert parsed_document.page_count == 548, (
        f"page_count 应为 548（PDF 实际页数），实际 {parsed_document.page_count}"
    )
    assert len(parsed_document.pages) == 548, (
        f"pages 条数应为 548，实际 {len(parsed_document.pages)}"
    )
    got = sorted(p.page for p in parsed_document.pages)
    assert got[0] == 1 and got[-1] == 548, f"页码范围应为 1..548，实际 {got[0]}..{got[-1]}"


def test_tables_extracted_and_paged(parsed_document):
    """表格必须真实提取，且 page/table_id 存在、页码落在文档范围内。"""
    tables = parsed_document.tables
    assert len(tables) >= 400, f"表格数应 ≥400（工单 6.1），实际 {len(tables)}"
    bad_id = [t.table_id for t in tables if not TABLE_ID_RE.match(t.table_id)]
    assert not bad_id, f"table_id 格式不合法（应形如 p152_t1）: {bad_id[:5]}"
    outside = [t.table_id for t in tables if not (1 <= t.page <= 548)]
    assert not outside, f"表格页码越界: {outside[:5]}"
    # table_id 内的页码必须与 page 字段一致
    mismatch = [t.table_id for t in tables if int(TABLE_ID_RE.match(t.table_id).group(1)) != t.page]
    assert not mismatch, f"table_id 页码与 page 字段不一致: {mismatch[:5]}"


def test_table_blocks_contain_markdown(parsed_document):
    """表格必须转成 Markdown 表结构（工单 6.1：表格结构化保留）。"""
    tables = parsed_document.tables
    no_md = [t.table_id for t in tables if "|" not in (t.markdown or "")]
    assert not no_md, f"以下表格缺少 Markdown 形式: {no_md[:5]}"
    no_sep = [t.table_id for t in tables if not MD_TABLE_RE.search(t.markdown or "")]
    assert len(no_sep) <= len(tables) * 0.02, (
        f"缺 Markdown 分隔行的表格过多（{len(no_sep)}/{len(tables)}），示例: {no_sep[:5]}"
    )
    rows_ok = [t for t in tables if t.rows]
    assert rows_ok, "所有表格的二维 rows 均为空，表格未真正结构化"


def test_cross_page_tables_merged(parsed_document):
    """跨页表格合并应生效（工单 6.1：跨页表格不丢行）。"""
    merged = [t for t in parsed_document.tables if t.merged_from]
    assert parsed_document.merged_table_count >= 1, (
        f"merged_table_count 应 ≥1，实际 {parsed_document.merged_table_count}"
    )
    assert len(merged) >= 1, "没有任何表格记录了 merged_from（跨页合并未落到数据上）"
    for t in merged[:5]:
        assert t.page_end >= t.page, f"{t.table_id} 的 page_end({t.page_end}) < page({t.page})"


def test_structured_blocks_have_required_fields(parsed_document):
    """结构化条目必须含 page/section/type/content（工单 6.1 硬要求）。"""
    blocks = parsed_document.blocks
    assert len(blocks) >= 10000, f"结构化块数应 ≥10000，实际 {len(blocks)}"
    sample = blocks[:2000]
    for blk in sample:
        assert isinstance(blk.page, int) and 1 <= blk.page <= 548, f"块页码非法: {blk.page}"
        assert blk.type in ("text", "table"), f"块类型非法: {blk.type}"
        assert isinstance(blk.content, str), "content 必须为字符串"
        assert isinstance(blk.section, str), "section 必须为字符串"
    empty = [b for b in sample if not b.content.strip()]
    assert len(empty) <= len(sample) * 0.01, f"空内容块过多: {len(empty)}/{len(sample)}"


def test_parser_reports_no_table_errors(parsed_document):
    """表格识别失败计数应为 0（实测值；非 0 即需在报告中如实说明）。"""
    assert parsed_document.table_errors == 0, (
        f"table_errors 应为 0，实际 {parsed_document.table_errors}"
    )


def test_ocr_not_faked(parsed_document):
    """文字版 PDF 必须显式标记跳过 OCR——不得伪造 OCR 能力（环境事实 2.2）。"""
    assert parsed_document.ocr_skipped is True, "文字版文档应显式 ocr_skipped=True"


def test_golden_evidence_pages_exist_in_document(parsed_document, golden, chunks):
    """每条 golden 的证据必须能落到真实页/块上。

    注意：**不使用** ``evidence_pages`` 作为唯一真值（存在错标，环境事实 §5.1），
    这里只校验「证据原文能在某个块里定位」，并如实打印每题的落点。
    """
    texts = {c.chunk_id: norm_b(c.content) for c in chunks}
    page_of = {c.chunk_id: c.page for c in chunks}
    located: dict[int, list[int]] = {}
    for item in golden:
        ev = norm_b(item.evidence)
        pages = sorted({page_of[cid] for cid, t in texts.items() if ev and ev in t})
        located[item.id] = pages
    # 8/10 题证据整段可定位；Q95（含「……」）与 Q207（合成引用串）为已知数据缺陷
    found = [qid for qid, pages in located.items() if pages]
    assert len(found) >= 8, f"整段证据可定位的题应 ≥8，实际 {len(found)}: 落点 {located}"
    assert 129 in located[260] and 129 in located[33], f"Q260/Q33 证据应在 p129: {located}"
    assert 152 in located[793] and 152 in located[34], f"Q793/Q34 证据应在 p152: {located}"
    # 页码必须落在文档范围内
    for qid, pages in located.items():
        for p in pages:
            assert 1 <= p <= parsed_document.page_count, f"Q{qid} 证据页越界: {p}"
