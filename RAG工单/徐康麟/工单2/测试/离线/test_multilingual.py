"""T3 离线测试 ⑥：中英文双语问答。

工单：人工智能NLP-RAG-基于PDF文档的问答系统优化
覆盖（设计/验收标准.md 验收 6、工单「中英文双语」）：

- 中文问中文答（``[页码: N]``）、英文问英文答（``[Page: N]``）；
- **英文问题能跨语言检索到中文证据**（``search_query`` 经语言桥接转为中文；
  引用的 chunk 正文为中文）；
- 至少 5 个英文问题。

英文问题取自工单1 ``rag_en`` 评估集同款 5 题（543/531/207/33/95），
便于与基线对照；本文件不改语料、不新增 golden 英文标准答案，
因此对英文只断言**语言/引用/跨语言召回**三类可客观判定的性质。
"""

from __future__ import annotations

import re

import pytest

#: 英文问题 → 对应工单问题编号（用于复用该题的证据块集合）
EN_QUESTIONS: tuple[tuple[int, str], ...] = (
    (543, "What is the registered capital of Wuhan Xingtu Xinke Electronics Co., Ltd.?"),
    (531, "Who is the legal representative of the company?"),
    (207, "How much of the raised funds will be used to supplement working capital?"),
    (95, "Which technical standard did the company participate in formulating?"),
    (33, "What percentage of main business revenue came from the military sector during the reporting period?"),
)

CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def test_language_detection():
    """语言判定必须准确（中英文切换的基础）。"""
    from app.core.language import detect_language, resolve_answer_language

    assert detect_language("注册资本是多少？") == "zh"
    assert detect_language("What is the registered capital?") == "en"
    assert resolve_answer_language("What is the registered capital?", "auto") == "en"
    assert resolve_answer_language("注册资本是多少？", "auto") == "zh"
    print("\n[语言] detect_language / resolve_answer_language 中英判定正确")


def test_chinese_question_chinese_answer(engine, query_understanding, golden):
    """中文提问：回答为中文、带 ``[页码: N]`` 引用、页码可回查。

    走端到端引擎（引用由编排层 ``CitationManager`` 附着，见 conftest.engine 说明）。
    """
    checked = 0
    for item in golden[:5]:
        analysis = query_understanding.analyze(item.question)
        assert analysis.language == "zh", f"Q{item.id} 中文提问被判成 {analysis.language}"
        answer = engine.ask(item.question)
        assert answer.answer.strip(), f"Q{item.id} 回答为空"
        assert answer.language == "zh", f"Q{item.id} 回答语言应为 zh，实际 {answer.language}"
        assert CJK_RE.search(answer.answer), f"Q{item.id} 中文回答不含中文: {answer.answer!r}"
        assert answer.citations, f"Q{item.id} 回答无引用（citations 为空）"
        assert "[页码:" in answer.answer, f"Q{item.id} 回答未带中文引用标签: {answer.answer!r}"
        for c in answer.citations:
            assert 1 <= c.page <= 548, f"Q{item.id} 引用页码越界: {c.page}"
            assert c.chunk_id, f"Q{item.id} 引用缺 chunk_id"
        checked += 1
        print(f"\n[中文] Q{item.id} mode={answer.mode} 引用页={[c.page for c in answer.citations]} "
              f"答案={answer.answer[:70]!r}")
    assert checked == 5, f"应校验 5 个中文问题，实际 {checked}"


@pytest.mark.slow
def test_english_questions_answered_in_english_with_real_citations(
        engine, retriever, query_understanding, evidence_hit_chunks):
    """英文提问：回答为英文、引用真实页码、且**跨语言召回中文证据**。

    引用走端到端引擎（编排层附着 citations）；跨语言召回走检索层直接校验。
    质量说明：实测英文抽取式答案为「英文模板 + 中文证据片段」形态
    （如 ``According to the prospectus: 第五节 发行人基本情况 [Page: 52]``），
    含较多中文——本用例断言「英文存在、引用真实、语言标记正确」，
    并**如实打印中文占比**供 T5/T7 评估英文作答质量，不掩饰该形态。
    """
    recalled = 0
    failures: list[str] = []
    print("\n[英文] 逐题结果：")
    for qid, question in EN_QUESTIONS:
        analysis = query_understanding.analyze(question)
        assert analysis.language == "en", f"{question!r} 未判为英文"
        # 语言桥接：检索用的 search_query 必须含中文（否则无法命中中文语料）
        assert CJK_RE.search(analysis.search_query or ""), (
            f"{question!r} 的 search_query 未桥接为中文: {analysis.search_query!r}"
        )
        contexts = retriever.retrieve_multi(query_understanding.search_queries(analysis), analysis)
        assert contexts, f"{question!r} 检索为空"
        cjk_ctx = [c for c in contexts if CJK_RE.search(c.content or "")]
        assert cjk_ctx, f"{question!r} 未召回到中文证据块"
        hit = any(c.chunk.chunk_id in evidence_hit_chunks[qid] for c in contexts)
        recalled += hit

        answer = engine.ask(question)
        cjk_ratio = len(CJK_RE.findall(answer.answer)) / max(1, len(answer.answer))
        print(f"   Q{qid} 证据召回={'✅' if hit else '❌'} 引用页={[c.page for c in answer.citations]} "
              f"is_unknown={answer.is_unknown}({answer.unknown_reason or '-'}) "
              f"中文占比={cjk_ratio:.0%} 答案={answer.answer[:70]!r}")
        # 逐条收集验收 6 的判据，最后统一断言（失败时一次给出全部问题）
        if answer.is_unknown:
            failures.append(f"Q{qid} 被拒答「{answer.answer}」(reason={answer.unknown_reason}, "
                            f"search_query={analysis.search_query!r})")
        elif answer.language != "en":
            failures.append(f"Q{qid} 回答语言={answer.language}，应为 en")
        elif not re.search(r"[A-Za-z]{3,}", answer.answer):
            failures.append(f"Q{qid} 回答不含英文: {answer.answer[:60]!r}")
        elif not answer.citations:
            failures.append(f"Q{qid} 回答无引用")
        else:
            for c in answer.citations:
                assert 1 <= c.page <= 548, f"{question!r} 引用页码越界: {c.page}"
                assert c.chunk_id, f"{question!r} 引用缺 chunk_id"
    assert recalled >= 4, f"英文问题跨语言召回证据应 ≥4/5，实际 {recalled}/5"
    print(f"[英文] 跨语言召回证据 = {recalled}/5")
    assert not failures, (
        f"验收 6（英文作答）未达标：{len(failures)}/{len(EN_QUESTIONS)} 个英文问题不合格。\n  - "
        + "\n  - ".join(failures)
    )


def test_english_citation_label_format():
    """英文引用标签必须是 ``[Page: N]``，中文为 ``[页码: N]``。"""
    from app.models.schemas import Citation

    c = Citation(page=52, chunk_id="c000158", snippet="x", section="", score=1.0)
    assert c.label("en") == "[Page: 52]", c.label("en")
    assert c.label("zh") == "[页码: 52]", c.label("zh")
    print(f"\n[引用标签] en={c.label('en')} zh={c.label('zh')}")


def test_english_answer_builder_uses_evidence():
    """英文抽取式答案生成器应基于证据文本产出英文答案（不伪造翻译能力）。"""
    from app.core.language import build_english_extractive_answer

    evidence = "公司目前已经成为军队视频指挥领域的重要供应商。"
    out = build_english_extractive_answer(evidence, 160)
    print(f"\n[英文生成] {out!r}")
    assert out and out.strip(), "英文抽取式答案为空"
    assert re.search(r"[A-Za-z]{3,}", out), "英文抽取式答案不含英文"
