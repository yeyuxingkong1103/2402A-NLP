# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化
tests/test_chunker_optimized.py —— 工单二语义分块器单元测试

覆盖：元数据完整性、父子链接、滑动窗口尺寸与重叠、语义边界（标题）、表格父块、页码归属
"""
import pytest

from src.chunker_optimized import (chunk_semantic, sliding_window,
                                   build_parent_blocks, build_children)

DOC_ID = "testdoc0000000001"


@pytest.fixture
def parsed_mini():
    """构造 mini 解析数据（Step 2 产物结构）：2 个标题 + 正文 + 1 张表格"""
    long_body = "。".join(f"这是第{i}句关于公司经营的完整描述" for i in range(80)) + "。"
    return {
        "doc_id": DOC_ID,
        "filename": "sample_opt.pdf",
        "layout_blocks": [
            {"page": 1, "type": "heading", "text": "第一节 公司概况", "level": 1, "y0": 60, "y1": 80, "font_size": 16},
            {"page": 1, "type": "body", "text": "公司主要从事电子信息系统的研发。", "y0": 100, "y1": 120},
            {"page": 1, "type": "body", "text": long_body, "y0": 130, "y1": 700},
            {"page": 2, "type": "heading", "text": "第二节 风险因素", "level": 1, "y0": 60, "y1": 80, "font_size": 16},
            {"page": 2, "type": "body", "text": "公司面临市场竞争风险和技术风险。", "y0": 100, "y1": 120},
            {"page": 1, "type": "header", "text": "测试公司 招股说明书 第1页", "y0": 10, "y1": 30},
        ],
        "tables_structured": [
            {"page": 2, "table_index": 1, "n_rows": 3, "n_cols": 2, "engine": "pdfplumber",
             "headers": ["项目", "金额"],
             "rows": [["营业收入", "100万"], ["净利润", "20万"]],
             "markdown": "| 项目 | 金额 |\n| --- | --- |\n| 营业收入 | 100万 |\n| 净利润 | 20万 |"},
        ],
        "headings": [],
        "pages": [], "text": "", "total_pages": 2,
    }


@pytest.fixture
def result(parsed_mini):
    return chunk_semantic(parsed_mini)


# ---------- 1. 元数据完整性（人工智能NLP-RAG-基于PDF文档的问答系统优化） ----------
def test_child_metadata_fields(result):
    for c in result["chunks"]:
        for key in ("page", "heading", "parent_id", "chunk_type", "role",
                    "chunk_id", "text", "char_count", "global_index"):
            assert key in c, f"子块缺少字段 {key}: {c['chunk_id']}"
        assert c["role"] == "child"


def test_parent_metadata_fields(result):
    for p in result["parent_chunks"]:
        for key in ("page", "heading", "chunk_type", "role", "chunk_id", "text"):
            assert key in p, f"父块缺少字段 {key}"
        assert p["role"] == "parent"


# ---------- 2. 父子链接（人工智能NLP-RAG-基于PDF文档的问答系统优化） ----------
def test_parent_child_link(result):
    parent_ids = {p["chunk_id"] for p in result["parent_chunks"]}
    assert all(c["parent_id"] in parent_ids for c in result["chunks"]), "子块 parent_id 必须指向存在的父块"
    assert result["total_child_chunks"] == len(result["chunks"])
    assert result["total_parent_chunks"] == len(result["parent_chunks"])


def test_every_parent_has_children(result):
    for p in result["parent_chunks"]:
        kids = [c for c in result["chunks"] if c["parent_id"] == p["chunk_id"]]
        assert kids, f"父块 {p['chunk_id']} 没有任何子块"


# ---------- 3. 滑动窗口 600/100（人工智能NLP-RAG-基于PDF文档的问答系统优化） ----------
def test_sliding_window_size_limit():
    text = "。".join(f"第{i}句完整语义描述内容较长用于测试窗口" for i in range(120)) + "。"
    pieces = sliding_window(text, 600, 100)
    assert len(pieces) >= 2
    for piece in pieces:
        assert len(piece) <= 610, f"子块超长: {len(piece)}"  # 标点对齐容差
    assert all(len(p) > 0 for p in pieces)


def test_sliding_window_overlap():
    text = "。".join(f"第{i}句语义连续内容重叠验证" for i in range(120)) + "。"
    pieces = sliding_window(text, 600, 100)
    a, b = pieces[0], pieces[1]
    tail = a[-100:]
    assert tail[:30] in b or b[:30] in a, "相邻子块应存在重叠"


def test_short_text_single_window():
    assert sliding_window("短文本", 600, 100) == ["短文本"]


# ---------- 4. 语义边界：标题层级（人工智能NLP-RAG-基于PDF文档的问答系统优化） ----------
def test_heading_starts_new_parent(result):
    ps = result["parent_chunks"]
    headings = [p["heading"] for p in ps if p["heading"]]
    assert "第一节 公司概况" in headings and "第二节 风险因素" in headings
    # 标题所在父块以标题文本开头
    first = ps[0]
    assert first["text"].startswith("第一节 公司概况")


def test_header_footer_excluded(result):
    joined = "".join(p["text"] for p in result["parent_chunks"])
    assert "招股说明书 第1页" not in joined, "页眉不应进入任何父块"


# ---------- 5. 表格父块（人工智能NLP-RAG-基于PDF文档的问答系统优化） ----------
def test_table_parent_block(result):
    table_parents = [p for p in result["parent_chunks"] if p["chunk_type"] == "table"]
    assert len(table_parents) == 1
    assert table_parents[0]["page"] == 2
    assert table_parents[0]["text"].startswith("| 项目 | 金额 |")


def test_table_child_type(result):
    table_children = [c for c in result["chunks"] if c["chunk_type"] == "table"]
    assert table_children, "表格应产生 table 类型子块"
    assert all(c["parent_id"].startswith(DOC_ID) for c in table_children)


# ---------- 6. 页码归属（人工智能NLP-RAG-基于PDF文档的问答系统优化） ----------
def test_page_metadata(result):
    p2_parents = [p for p in result["parent_chunks"] if p["page"] == 2]
    assert p2_parents, "第2页应产生父块"
    for c in result["chunks"]:
        assert isinstance(c["page"], int) and c["page"] in (1, 2)


def test_build_children_inheritance(parsed_mini):
    parents = build_parent_blocks(parsed_mini)
    children = build_children(parents)
    by_id = {p["chunk_id"]: p for p in parents}
    for c in children[:20]:
        parent = by_id[c["parent_id"]]
        assert c["page"] == parent["page"] and c["heading"] == parent["heading"]
