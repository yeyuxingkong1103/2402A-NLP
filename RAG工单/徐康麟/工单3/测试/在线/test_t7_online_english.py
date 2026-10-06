# -*- coding: utf-8 -*-
"""在线级：英文问答覆盖（7 条）+ 允许的降级留痕。

工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化

背景：t17/t19/t20 窗口曾出现「英文提问被拒答」回归（根因 = 可答性闸门的 ``keyword_coverage`` 是词面
token 覆盖率，英文 query 与中文语料天然无重叠）。t21/t22 修复后，英文侧口径为：

    * **不得拒答**（0 条「不清楚」）；
    * **必须带可回溯引用**（引用由 §2.4 独立判，与语言无关）；
    * ``language == "en"`` = 英文正文**达标**；
    * ``language != "en"`` 时**必须**有 ``language_degraded`` 降级留痕（WARNING 事件）→ 记为**允许的降级**，
      **不得**当作「英文正文达标」计入，也不得静默降级。

本文件只报实跑数字：达标数、降级数、拒答数各自留痕。
"""

from __future__ import annotations

from typing import Any

import pytest

from common import assertions, paths
from common.reports import trace_events, write_report

pytestmark = [pytest.mark.online, pytest.mark.slow]

#: 7 条英文问题（覆盖 PDF1 字段题/数值题/文本题 + PDF2 发行与关联方题）
ENGLISH_QUESTIONS: tuple[tuple[str, str], ...] = (
    ("EN1", "What is the registered capital of Wuhan Xingtu Xinke Electronics Co., Ltd.?"),
    ("EN2", "Who is the legal representative of Wuhan Xingtu Xinke Electronics Co., Ltd.?"),
    ("EN3", "What were the company's revenues from the military domain during the reporting period?"),
    ("EN4", "According to the prospectus, what are the upstream industries of the electronic information industry?"),
    ("EN5", "In which domain has Wuhan Xingtu Xinke Electronics Co., Ltd. become an important supplier?"),
    ("EN6", "What is the number of shares issued by Wuhan P&S Information Technology Co., Ltd.?"),
    ("EN7", "Which related-party enterprises are not controlled by Wuhan P&S Information Technology Co., Ltd.?"),
)
#: 允许的降级留痕关键字（产品在降级时必须写这个事件）
DEGRADE_MARKERS = ("language_degraded", "keep_chinese")


@pytest.fixture(scope="session")
def english_results(engine: Any, page_lookup: Any, page_counts: dict[str, int],
                    chunk_lookup: dict[str, Any]) -> list[dict[str, Any]]:
    """跑 7 条英文问题，逐条记录语言/拒答/引用可回溯/降级留痕。"""
    from common import artifacts  # noqa: PLC0415

    files = list(page_counts)
    chunk_map = chunk_lookup if chunk_lookup else {r["chunk_id"]: r for r in artifacts.load_chunks()}
    offsets = {name: len(trace_events(name)) for name in ("app.log", "rag_trace.jsonl")}
    rows: list[dict[str, Any]] = []
    for tag, question in ENGLISH_QUESTIONS:
        answer = engine.ask(question, session_id=f"t8-en-{tag}", stream=False)
        citations = list(getattr(answer, "citations", []) or [])
        cite_rows = []
        for cite in citations:
            check = assertions.check_citation_traceable(
                cite, page_lookup=page_lookup, page_counts=page_counts,
                discovered_files=files, chunk_lookup=chunk_map)
            cite_rows.append({"file_name": getattr(cite, "file_name", ""),
                              "page": getattr(cite, "page", None),
                              "chunk_id": getattr(cite, "chunk_id", ""), "ok": check.ok,
                              "detail": check.detail})
        after = {name: trace_events(name)[offsets[name]:] for name in offsets}
        degrade_events = [str(row.get("event", "")) for rows_ in after.values() for row in rows_
                          if any(marker in str(row.get("event", "")) or marker in str(row.get("message", ""))
                                 for marker in DEGRADE_MARKERS)]
        rows.append({
            "tag": tag, "question": question,
            "language": str(getattr(answer, "language", "")),
            "is_unknown": bool(getattr(answer, "is_unknown", False)),
            "unknown_reason": getattr(answer, "unknown_reason", None),
            "answer_text": str(answer.text), "first_token_ms": float(answer.first_token_ms or 0.0),
            "citations": cite_rows,
            "citation_ok": bool(cite_rows) and all(c["ok"] for c in cite_rows),
            "degrade_events": sorted(set(degrade_events)),
            "english_body": str(getattr(answer, "language", "")) == "en",
            "allowed_degrade": (str(getattr(answer, "language", "")) != "en") and bool(degrade_events),
        })
    write_report("online_english", "英文问答覆盖（7 条，含允许的降级留痕）", rows,
                 extra={"达标(language=en)": sum(1 for r in rows if r["english_body"]),
                        "允许的降级": sum(1 for r in rows if r["allowed_degrade"]),
                        "拒答数": sum(1 for r in rows if r["is_unknown"])})
    return rows


def test_no_english_refusal(english_results: list[dict[str, Any]]) -> None:
    """英文侧**不得拒答**（0 条「不清楚」）—— t21 修复的回归点。"""
    refused = [(r["tag"], r["unknown_reason"], r["answer_text"][:40])
               for r in english_results if r["is_unknown"]]
    assert not refused, (f"英文提问被拒答（回归未修复）：{refused}"
                         f"【根因参考：可答性闸门 keyword_coverage 缺跨语言适配】")


def test_english_answers_have_traceable_citations(english_results: list[dict[str, Any]]) -> None:
    """英文答案必须带**可回溯**引用（引用与语言无关，1-based 物理页 + chunk 回查一致）。"""
    problems = [(r["tag"], r["citations"]) for r in english_results if not r["citation_ok"]]
    assert not problems, f"英文答案缺少可回溯引用：{problems}"


def test_language_is_en_or_explicitly_degraded(english_results: list[dict[str, Any]]) -> None:
    """语言口径：``language=en`` 为达标；否则**必须**有 ``language_degraded`` 留痕（允许的降级）。

    禁止「静默降级」：既不标 en、也没有降级事件的，一律判失败。
    """
    silent = [r["tag"] for r in english_results
              if not r["english_body"] and not r["allowed_degrade"]]
    assert not silent, (f"英文侧出现静默降级（未标 en 且无 language_degraded 留痕）：{silent}；"
                        f"逐条见 测试/留痕/online_english.json")


def test_english_first_token_budget(english_results: list[dict[str, Any]]) -> None:
    """英文题同样受**逐题**首字 ≤3000 ms 约束（拒答题 first_token 为 0，属例外并已被上一条拦住）。"""
    answered = {f"EN-{r['tag']}": r["first_token_ms"] for r in english_results if not r["is_unknown"]}
    checks = assertions.check_first_token(answered) if answered else []
    bad = [c for c in checks if not c.ok]
    assert not bad, "\n".join(c.render() for c in bad)
