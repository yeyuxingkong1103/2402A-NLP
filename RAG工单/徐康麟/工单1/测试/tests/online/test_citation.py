"""在线测试：引用真实性（工单 9.2 / 5.6 / 验收标准 3）。

测试目标：
1. 答案中的引用页码必须**真实存在于 PDF 范围内**（1..548，用 PyMuPDF 实测页数校验）；
2. 引用携带的 ``chunk_id`` 必须能在 SQLite 中查到，且页码与片段一致；
3. 引用摘要必须来自原文片段（禁止编造引用）；
4. 引用标签格式为 ``[页码: N]``，且引用页码必须来自本次检索命中的页码；
5. 越界引用（模型幻觉）必须能被 ``CitationManager.validate`` 识别。

索引缺失时整个模块跳过：
``索引未就绪：请先运行 python scripts/build_index.py``。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

# 目录结构：<root>/测试/tests/<子目录>/xxx.py
#   parents[0]=子目录 parents[1]=tests parents[2]=测试 parents[3]=项目根
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = PROJECT_ROOT / "研发"
GOLDEN_PATH = PROJECT_ROOT / "data" / "eval" / "golden_qa.jsonl"
LABEL_PATTERN = re.compile(r"^\[页码: \d+\]$")


def _load_golden_items():
    """读取标准答案文件（缺失时返回空列表，由固件负责跳过）。"""
    from app.models.schemas import GoldenQA

    if not GOLDEN_PATH.exists():
        return []
    return [
        GoldenQA(**json.loads(line))
        for line in GOLDEN_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def pytest_generate_tests(metafunc) -> None:
    """把工单 10 题参数化到引用校验用例上。"""
    if "golden_item" not in metafunc.fixturenames:
        return
    metafunc.parametrize(
        "golden_item",
        [pytest.param(item, id=f"q{item.id}") for item in _load_golden_items()],
    )


# --------------------------------------------------------------------------
# 1. 页码真实性
# --------------------------------------------------------------------------
def test_citation_pages_within_real_pdf_range(engine, golden_item, pdf_page_count, requires_index) -> None:
    """引用页码必须落在 PDF 真实页数范围内（越界即为编造引用）。"""
    answer = engine.ask(golden_item.question, save=False)
    assert answer.citations, f"题号 {golden_item.id} 的答案没有引用来源"

    for citation in answer.citations:
        assert 1 <= citation.page <= pdf_page_count, (
            f"题号 {golden_item.id} 的引用页码 {citation.page} 越界（PDF 共 {pdf_page_count} 页）"
        )


def test_citation_pages_come_from_retrieved_chunks(engine, golden_item, requires_index) -> None:
    """引用页码必须来自本次检索命中的片段，不得引用未命中的页。"""
    answer = engine.ask(golden_item.question, save=False)
    assert answer.pages, f"题号 {golden_item.id} 未记录检索命中页码"
    cited = {citation.page for citation in answer.citations}
    assert cited <= set(answer.pages), (
        f"题号 {golden_item.id} 引用了未命中的页码：{sorted(cited - set(answer.pages))}"
    )


# --------------------------------------------------------------------------
# 2. chunk_id 可追溯
# --------------------------------------------------------------------------
def test_citation_chunk_ids_exist_in_sqlite(engine, golden_item, requires_index) -> None:
    """每个引用的 chunk_id 都必须能在 SQLite 中查到，且页码一致。"""
    answer = engine.ask(golden_item.question, save=False)
    for citation in answer.citations:
        assert citation.chunk_id, f"题号 {golden_item.id} 的引用缺少 chunk_id，前端无法展开原文"
        chunk = engine.store.get_chunk(citation.chunk_id)
        assert chunk is not None, f"引用片段 {citation.chunk_id} 在 SQLite 中不存在（疑似编造）"
        assert chunk.page == citation.page, (
            f"引用页码 {citation.page} 与片段 {citation.chunk_id} 的实际页码 {chunk.page} 不一致"
        )
        assert chunk.doc_id == engine.doc_id, f"引用片段 {citation.chunk_id} 不属于当前文档"


def test_citation_snippet_comes_from_original_text(engine, golden_item, requires_index) -> None:
    """引用摘要必须来自原文片段，用于前端"展开查看原文"。"""
    answer = engine.ask(golden_item.question, save=False)
    for citation in answer.citations:
        chunk = engine.store.get_chunk(citation.chunk_id)
        assert chunk is not None, f"引用片段 {citation.chunk_id} 不存在"
        snippet = citation.snippet.rstrip("…").strip()
        assert snippet, "引用摘要为空"
        assert snippet[:40] in chunk.content, (
            f"引用摘要与原文不符，疑似编造：摘要={snippet[:40]!r}"
        )


# --------------------------------------------------------------------------
# 3. 标签格式与整体校验
# --------------------------------------------------------------------------
def test_citation_labels_use_required_format(engine, golden_item, requires_index) -> None:
    """引用标签必须形如 ``[页码: N]``（工单 5.5 规定的格式）。"""
    answer = engine.ask(golden_item.question, save=False)
    for citation in answer.citations:
        assert LABEL_PATTERN.match(citation.label()), f"引用标签格式错误：{citation.label()!r}"


def test_citation_validation_ratio_is_full(engine, golden_item, requires_index) -> None:
    """引用校验结果：合法率必须为 100%，且没有越界页码。"""
    answer = engine.ask(golden_item.question, save=False)
    validation = engine.citation.validate(answer)
    assert validation["total"] == len(answer.citations), "校验统计的引用总数与答案不一致"
    assert validation["invalid_pages"] == [], f"存在越界引用：{validation['invalid_pages']}"
    assert validation["valid_ratio"] == 1.0, f"引用合法率应为 1.0，实际 {validation['valid_ratio']}"
    assert validation["has_snippet"] is True, "引用必须携带原文片段"


def test_render_markdown_lists_all_citations(engine, requires_index) -> None:
    """Markdown 渲染必须列出全部引用（页码 + chunk_id + 原文摘要）。"""
    answer = engine.ask("武汉兴图新科电子股份有限公司注册资本是多少？", save=False)
    markdown = engine.citation.render_markdown(answer)

    assert "**引用来源**" in markdown, "渲染结果缺少引用来源小节"
    for citation in answer.citations:
        assert citation.label() in markdown, f"渲染结果缺少引用 {citation.label()}"
        assert citation.chunk_id in markdown, f"渲染结果缺少 chunk_id {citation.chunk_id}"


def test_out_of_range_citation_is_detected(engine, requires_index) -> None:
    """引用管理器必须使用当前文档的真实页码范围，能识别越界引用。"""
    from app.models.schemas import Answer, Citation

    assert engine.citation.valid_pages, "引用管理器未装载文档页码范围"
    assert max(engine.citation.valid_pages) == 548, (
        f"文档页码范围上限应为 548，实际 {max(engine.citation.valid_pages)}"
    )

    fake = Answer(answer="见招股书 [页码: 9999]", citations=[Citation(page=9999, chunk_id="", snippet="")])
    validation = engine.citation.validate(fake)
    assert validation["invalid_pages"] == [9999], f"越界引用未被识别：{validation}"
    assert validation["valid_ratio"] == 0.0, "全部引用越界时合法率应为 0"
