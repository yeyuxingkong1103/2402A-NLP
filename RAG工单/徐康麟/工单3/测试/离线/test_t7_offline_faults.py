# -*- coding: utf-8 -*-
"""离线级：故障容错（解析失败 / 检索为空）。

工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化

要求（工单硬要求）：**禁止静默失败** —— ``except`` 必须 log + 传播，或 log + 显式降级；
本用例断言「坏了要说话」：解析不存在的文件与损坏文件必须抛 ``PdfParseError``，
空检索必须让生成侧回「不清楚」并给出枚举内的 ``unknown_reason``，且都不得静默返回空结果。
"""

from __future__ import annotations

from typing import Any

import pytest

from common import assertions, paths

pytestmark = [pytest.mark.offline, pytest.mark.linkage]


def test_missing_pdf_raises_explicitly() -> None:
    """不存在的 PDF：必须**显式抛错**（带错误码），不得返回空结果或静默继续。

    留痕说明（非缺陷，供 T11 参考）：``parse_pdf`` 的 ``_as_source()`` 位于 ``log.enter`` 之前，
    因此这条「前置校验」路径只会传播异常、不会写 ``func.error``；而真正进入解析流程后失败
    （损坏文件、页数异常）会写 ERROR 日志（见下一个用例）。
    """
    paths.ensure_dev_on_path()
    from app.core import pdf_parser  # noqa: PLC0415
    from app.core.errors import PdfParseError  # noqa: PLC0415

    ghost = paths.RAW_DIR / "不存在的语料.pdf"
    assert not ghost.exists()

    with pytest.raises(PdfParseError) as excinfo:
        pdf_parser.parse_pdf(ghost)
    error = excinfo.value
    assert "不存在" in str(error) or "打不开" in str(error), str(error)
    assert str(getattr(error, "code", "") or "").startswith("RAG-"), \
        f"前置校验失败必须带错误码，实测 code={getattr(error, 'code', None)!r}"
    assert ghost.name in str(error), f"错误信息应含文件名便于定位，实测 {error}"


def test_corrupt_pdf_raises_and_logs() -> None:
    """损坏文件（非 PDF 内容）：必须 **log + 传播**（``pdf.open_failed`` ERROR + ``PdfParseError``）。

    说明：不用 pytest 的 ``tmp_path``（本机沙箱禁止列临时目录），改写到 ``测试/留痕/``。
    """
    paths.ensure_dev_on_path()
    from app.core import pdf_parser  # noqa: PLC0415
    from app.core.errors import PdfParseError  # noqa: PLC0415

    corrupt = paths.ensure_trace_dir() / "corrupt_fixture.pdf"
    corrupt.write_bytes(b"this is definitely not a pdf file\n" * 20)
    watched = [paths.trace_file("app.log"), paths.trace_file("error.log")]
    offsets = {str(p): (p.stat().st_size if p.exists() else 0) for p in watched}
    try:
        with pytest.raises(PdfParseError):
            pdf_parser.parse_pdf(corrupt)
        fresh: list[str] = []
        for path in watched:
            if not path.exists() or path.stat().st_size <= offsets[str(path)]:
                continue
            with path.open("rb") as handle:
                handle.seek(offsets[str(path)])
                fresh.append(handle.read().decode("utf-8", errors="replace"))
    finally:
        corrupt.unlink(missing_ok=True)

    seen = "\n".join(fresh)
    assert seen.strip(), "损坏文件解析失败**没有任何新增日志** → 静默失败"
    assert "pdf.open_failed" in seen or "PdfParseError" in seen, \
        "损坏文件的失败没有写 ERROR 事件（应含 pdf.open_failed / PdfParseError）"


def test_pdf_page_count_matches_real_files(discovered_pdfs: list[Any]) -> None:
    """``pdf_page_count`` 与真实页数一致（1 号 548 / 2 号 350），防「读不到就当 0 页」。"""
    paths.ensure_dev_on_path()
    from app.core import pdf_parser  # noqa: PLC0415
    from common import pdf_probe  # noqa: PLC0415

    for path in discovered_pdfs:
        product = int(pdf_parser.pdf_page_count(path))
        direct = int(pdf_probe.page_count(str(path)))
        assert product == direct, f"{path.name} 页数不一致：产品 {product} vs pymupdf {direct}"
    counts = {p.name: int(pdf_parser.pdf_page_count(p)) for p in discovered_pdfs}
    assert 548 in counts.values(), f"应包含 548 页的 1 号 PDF：{counts}"
    assert 350 in counts.values(), f"应包含 350 页的 2 号 PDF：{counts}"


def test_empty_retrieval_yields_unknown(generator: Any, app_config: Any) -> None:
    """检索为空：生成侧必须返回「不清楚」（枚举内 reason）+ 无引用，不得编造答案。"""
    paths.ensure_dev_on_path()
    from app.core.retriever import RetrievalResult  # noqa: PLC0415

    empty = RetrievalResult(query="空检索注入", normalized_query="空检索注入", rewritten_query=None,
                            file_names=None, top_k=int(app_config.retrieval.top_k), chunks=[],
                            stages={}, candidates_count=0, table_fallback_used=False,
                            trace_id="t8-empty-retrieval")
    answer = generator.answer("武汉兴图新科电子股份有限公司注册资本是多少？", empty,
                              trace_id="t8-empty-retrieval")
    assert bool(answer.is_unknown) is True, f"空检索却给出答案：{answer.text[:80]!r}"
    assert answer.text.strip() == app_config.answer.unknown_text, f"拒答文本应为「不清楚」，实测 {answer.text!r}"
    assert not list(answer.citations), "拒答不得带引用"
    # 枚举以 设计/接口设计.md §3.19 冻结取值为准（empty_retrieval/low_score/low_coverage/…）
    assert answer.unknown_reason in assertions.UNKNOWN_REASONS, \
        f"reason={answer.unknown_reason!r} 不在冻结枚举内：{assertions.ANSWERABILITY_REASONS}"
    assert answer.unknown_reason == "empty_retrieval", \
        f"空检索应报 empty_retrieval，实测 {answer.unknown_reason!r}"
