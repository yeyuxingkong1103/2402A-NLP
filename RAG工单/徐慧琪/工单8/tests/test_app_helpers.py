# 工单编号：人工智能NLP-RAG-基于GraphRAG实现金融问答
from src.ui_helpers import (answer_with_sources, render_answer_md,
                            render_facts_md, render_hits_md)

RESULT = {
    "question": "平安银行不良贷款率",
    "answer": "平安银行2019年末不良贷款率为1.65%。",
    "mode": "metric_compare", "lang": "zh", "degraded": False,
    "degraded_reason": "",
    "graph_facts": ["平安银行2019年末的不良贷款率为 1.65%（来源：a.pdf 第 23 页）"],
    "text_hits": [{"chunk_id": "c1", "source": "a.pdf", "page": 23,
                   "score": 0.9, "text": "不良贷款率1.65%"}],
    "timings": {"total": 1.8},
}


def test_render_answer_md_contains_answer():
    md = render_answer_md(RESULT, "zh")
    assert "1.65%" in md


def test_render_answer_md_marks_degraded():
    r = dict(RESULT, degraded=True, degraded_reason="LLM 生成失败：RuntimeError")
    md = render_answer_md(r, "zh")
    assert "降级" in md
    assert "LLM 生成失败" in md


def test_render_facts_md_lists_all():
    md = render_facts_md(RESULT["graph_facts"])
    assert "1.65%" in md
    assert md.count("\n") >= 1


def test_render_facts_md_empty():
    assert "未检索到" in render_facts_md([])


def test_render_hits_md_shows_page():
    md = render_hits_md(RESULT["text_hits"])
    assert "a.pdf" in md and "23" in md


def test_render_hits_md_empty():
    assert "未检索到" in render_hits_md([])


def test_answer_with_sources_appends_page():
    text = answer_with_sources(RESULT)
    assert "1.65%" in text
    assert "a.pdf" in text


def test_answer_with_sources_without_hits():
    r = dict(RESULT, text_hits=[])
    assert answer_with_sources(r)
