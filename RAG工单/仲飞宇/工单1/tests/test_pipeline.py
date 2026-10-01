# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
# 工单01 - 基于PDF文档的问答系统
"""
冒烟测试：不依赖 Milvus，覆盖解析 / 分块 / 去重 / 规则命中 / 文本规范化。

运行：pytest tests/ -v
Milvus 相关用例在不可达时自动跳过（skip），保证「没起向量库也能跑测试」。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402
from app.core.chunker import chunk_pages  # noqa: E402
from app.core.dedup import content_hash, hamming, simhash  # noqa: E402
from app.core.evaluator import rule_hit  # noqa: E402
from app.core.pdf_parser import normalize_text, page_label, parse_pdf_full  # noqa: E402
from app.core.vectorstore import lexical_sparse  # noqa: E402

PDF = settings.data_path / "raw" / "招股说明书1.pdf"
needs_pdf = pytest.mark.skipif(not PDF.exists(), reason="招股说明书1.pdf 不存在")


# ----------------------------------------------------------------------
# 文本规范化
# ----------------------------------------------------------------------
def test_normalize_removes_cjk_spacing():
    # 实测 PDF 抽出来的样子
    assert normalize_text("注册资本5,520.00 万元") == "注册资本5,520.00万元"
    assert normalize_text("2019 年12 月16 日") == "2019年12月16日"
    assert normalize_text("3.55 元/ 股") == "3.55元/股"


def test_normalize_keeps_english_spacing():
    # 纯英文词组不能被压掉空格
    s = normalize_text("Wuhan Xingtu Xinke Electronics Co.,Ltd.")
    assert s == "Wuhan Xingtu Xinke Electronics Co.,Ltd."


def test_page_label_offset_zero():
    # 实测偏移为 0：index N ↔ 页脚 1-1-N
    assert page_label(22, 0) == "1-1-22"
    assert page_label(129, 0) == "1-1-129"


# ----------------------------------------------------------------------
# 规则化数值命中
# ----------------------------------------------------------------------
def test_rule_hit_all():
    ans = "分别为6,464.51万元、14,414.16万元、18,780.67万元和4,627.14万元。"
    ok, detail = rule_hit(ans, ["6,464.51", "14,414.16", "18,780.67", "4,627.14"], "all")
    assert ok, detail


def test_rule_hit_tolerates_comma_style():
    # 模型把 15,000.00 写成 15000.00 或 15000 也应算命中
    ok, _ = rule_hit("公司拟投入15000万元用于补充流动资金", ["15,000.00"], "any")
    assert ok


def test_rule_hit_missing():
    ok, detail = rule_hit("资料中未提及相关信息。", ["程家明"], "any")
    assert not ok and "缺失" in detail


def test_rule_hit_all_partial_is_fail():
    ok, _ = rule_hit("收入为6,464.51万元", ["6,464.51", "14,414.16"], "all")
    assert not ok, "all 模式下缺一个就应判未命中"


# ----------------------------------------------------------------------
# 去重
# ----------------------------------------------------------------------
def test_content_hash_stable():
    assert content_hash("abc  def") == content_hash("abcdef")


def test_simhash_similar_texts_are_close():
    a = "公司目前已经成为军队视频指挥领域的重要供应商，参与制定了全军第一个视频指挥系统技术标准。"
    b = "公司目前已经成为军队视频指挥领域的核心供应商，参与制定了全军第一个视频指挥系统技术标准。"
    c = "报告期内，公司来自军用领域的收入分别为6,464.51万元、14,414.16万元。"
    assert hamming(simhash(a), simhash(b)) < hamming(simhash(a), simhash(c))


# ----------------------------------------------------------------------
# 稀疏向量
# ----------------------------------------------------------------------
def test_lexical_sparse_keeps_numbers():
    """数字必须整体保留 —— 招股书的核心答案大量是数字。"""
    v = lexical_sparse("注册资本5,520.00万元，页码1-1-42")
    assert len(v) > 0
    # 索引是哈希后的整数，只验证非空与取值范围
    assert all(isinstance(k, int) and 0 <= k < 2**31 for k in v)


# ----------------------------------------------------------------------
# 端到端：解析 + 分块（需要 PDF）
# ----------------------------------------------------------------------
@needs_pdf
def test_parse_removes_header_footer_exactly_once_per_page():
    pages, stats = parse_pdf_full(PDF, offset=settings.page_label_offset, limit=40)
    # n_pages 是文档总页数（用于报告），本次实际解析的是 limit 页
    assert stats.n_pages == 548
    assert len(pages) == 40
    # 实测每页恰好一个页眉、一个页脚
    assert stats.n_header_removed == 40
    assert stats.n_footer_removed == 40


@needs_pdf
def test_key_page_21_has_registered_capital_and_legal_rep():
    """p21 是「法定代表人」「注册资本」两题的答案页 —— 解析错则两题必挂。"""
    pages, _ = parse_pdf_full(PDF, offset=settings.page_label_offset, limit=25)
    pc = pages[21]
    blob = "\n".join(t.text for t in pc.texts)
    blob += "\n" + "\n".join(tb.to_markdown() for tb in pc.tables)
    assert "程家明" in blob
    assert "5,520.00" in blob or "5,520" in blob


@needs_pdf
def test_key_page_29_has_supplementary_working_capital():
    """p29 的募集资金表是 id=207 的答案页。"""
    pages, _ = parse_pdf_full(PDF, offset=settings.page_label_offset, limit=35)
    pc = pages[29]
    blob = "\n".join(tb.to_markdown() for tb in pc.tables)
    assert "补充流动资金" in blob
    assert "15,000.00" in blob


@needs_pdf
def test_golden_sentence_survives_chunking():
    """
    p128 那句 137 字同时装着 id=260 与 id=33 两个答案，
    固定字数切块会把它切断 —— 本用例锁死这个回归。
    """
    pages, _ = parse_pdf_full(PDF, offset=settings.page_label_offset, limit=135)
    chunks = [c for c in chunk_pages(pages) if c.chunk_type == "text"]
    key = ("公司来自军用领域的收入分别为6,464.51万元、14,414.16万元、"
           "18,780.67万元和4,627.14万元，占主营业务收入比重分别为"
           "82.10%、97.31%、94.84%和94.34%")
    assert any(key in c.content for c in chunks), "金句被切断了"


@needs_pdf
def test_chunk_sizes_within_bounds():
    pages, _ = parse_pdf_full(PDF, offset=settings.page_label_offset, limit=80)
    texts = [c for c in chunk_pages(pages) if c.chunk_type == "text"]
    assert texts
    # 上限 600 字，允许个别无标点长段落到 800（chunker 有硬切兜底）
    assert max(c.n_chars for c in texts) <= 800


@needs_pdf
def test_table_chunk_is_markdown_with_title():
    pages, _ = parse_pdf_full(PDF, offset=settings.page_label_offset, limit=35)
    tables = [c for c in chunk_pages(pages) if c.chunk_type == "table"]
    assert tables
    md = tables[0].content
    assert "|" in md and "---" in md


# ----------------------------------------------------------------------
# Milvus（不可达则跳过）
# ----------------------------------------------------------------------
def test_milvus_roundtrip():
    from app.core.vectorstore import VectorStore, VectorStoreError
    store = VectorStore()
    ok, msg = store.health()
    if not ok:
        pytest.skip(f"Milvus 不可达：{msg[:80]}")
    assert store.collection
