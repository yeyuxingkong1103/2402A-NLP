# 工单编号：人工智能NLP-RAG-基于GraphRAG实现金融问答
from src.kg_select import score_chunk, select_chunks

CH_GOV = {"chunk_id": "g", "text": "公司控股股东为中国平安保险（集团）股份有限公司。",
          "source": "a.pdf", "page_idx": 30, "heading_path": ["公司治理"],
          "block_type": "text"}
CH_TOC = {"chunk_id": "t", "text": "目录 第一节 释义 第二节 公司简介",
          "source": "a.pdf", "page_idx": 1, "heading_path": ["目录"],
          "block_type": "text"}
CH_PLAIN = {"chunk_id": "p", "text": "本行坚持稳健经营。",
            "source": "a.pdf", "page_idx": 50, "heading_path": ["业务概要"],
            "block_type": "text"}


def test_whitelisted_section_scores_higher_than_plain():
    s_gov, _ = score_chunk(CH_GOV)
    s_plain, _ = score_chunk(CH_PLAIN)
    assert s_gov > s_plain


def test_trigger_words_raise_score():
    s1, r1 = score_chunk(CH_GOV)
    assert any("触发词" in r or "控股" in r for r in r1), r1


def test_toc_excluded():
    score, reasons = score_chunk(CH_TOC)
    assert score < 0
    assert any("目录" in r for r in reasons)


def test_select_chunks_sorted_desc():
    out = select_chunks([CH_TOC, CH_PLAIN, CH_GOV])
    assert [x["chunk"]["chunk_id"] for x in out][0] == "g"
    assert all(x["score"] >= 0 for x in out), "目录类应被过滤掉"


def test_select_chunks_respects_max_blocks():
    many = [dict(CH_GOV, chunk_id=f"g{i}") for i in range(50)]
    out = select_chunks(many, max_blocks=10)
    assert len(out) == 10


def test_select_chunks_returns_reasons():
    out = select_chunks([CH_GOV])
    assert out[0]["reasons"], "必须给出入选原因，便于人工核验"
