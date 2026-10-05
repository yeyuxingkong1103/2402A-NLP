# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
模块：table_parser / table_chunker 单元测试

覆盖工单3 的表格链路关键行为：
  - HTML → 单元格矩阵（含 rowspan/colspan 展开）
  - 表头识别（含"日期表头不能误判为数据行"这一实测踩坑）
  - 字段-值表（kv 模式）识别：2 列表 / 释义表
  - 行级描述与表头块文本格式
  - 质量分与低质量判定
  - 表格 chunk 的元数据完整性（工单硬要求 8 项）
  - 降级路径（无 HTML / 解析失败）
"""
from __future__ import annotations

import pytest

from src import table_chunker, table_parser
from src.table_parser import TableStruct

# --- 样例表格（取自招股说明书的真实形态，已脱敏处理格式）--------------------

HTML_KV_2COL = """<table>
<tr><td rowspan=1 colspan=1>发行股票类型</td><td rowspan=1 colspan=1>人民币普通股（A股）</td></tr>
<tr><td rowspan=1 colspan=1>发行股数</td><td rowspan=1 colspan=1>1,670万股</td></tr>
<tr><td rowspan=1 colspan=1>发行后总股本</td><td rowspan=1 colspan=1>6,670万股</td></tr>
</table>"""

HTML_HEADER_4COL = """<table>
<tr><td>项目</td><td>2019年7-9月</td><td>2018年7-9月</td><td>同比变动</td></tr>
<tr><td>营业收入</td><td>1,918.21</td><td>625.12</td><td>206.85%</td></tr>
<tr><td>营业利润</td><td>-226.29</td><td>-1,315.01</td><td></td></tr>
</table>"""

HTML_HEADER_DATE = """<table>
<tr><td>项目</td><td>2010-6-30</td><td>2009-12-31</td></tr>
<tr><td>流动资产</td><td>147,346,914.80</td><td>136,742,643.93</td></tr>
<tr><td>非流动资产</td><td>9,312,942.74</td><td>8,875,846.18</td></tr>
</table>"""

HTML_KV_DEFINITION = """<table>
<tr><td>发行人、兴图新科</td><td>指</td><td>武汉兴图新科电子股份有限公司</td></tr>
<tr><td>兴图新科有限</td><td>指</td><td>武汉兴图新科电子有限公司，发行人前身</td></tr>
<tr><td>华创兴图</td><td>指</td><td>北京华创兴图电子科技有限公司</td></tr>
</table>"""

HTML_ROWSPAN = """<table>
<tr><td rowspan=2>项目</td><td>2019年度</td><td>2018年度</td></tr>
<tr><td>金额</td><td>金额</td></tr>
<tr><td>营业收入</td><td>1,918.21</td><td>625.12</td></tr>
</table>"""


# ---------------------------------------------------------------------------
# HTML → 矩阵
# ---------------------------------------------------------------------------
def test_html_to_grid_basic():
    grid, n_cells = table_parser.html_to_grid(HTML_KV_2COL)
    assert len(grid) == 3
    assert all(len(r) == 2 for r in grid)
    assert grid[1] == ["发行股数", "1,670万股"]
    assert n_cells == 6


def test_html_to_grid_expands_colspan():
    html = ("<table><tr><td colspan=2>合计</td></tr>"
            "<tr><td>1</td><td>2</td></tr></table>")
    grid, _ = table_parser.html_to_grid(html)
    assert grid[0] == ["合计", "合计"]      # colspan 值向覆盖区域填充
    assert len(grid[0]) == len(grid[1]) == 2


def test_html_to_grid_expands_rowspan():
    grid, _ = table_parser.html_to_grid(HTML_ROWSPAN)
    assert grid[1][0] == "项目"             # rowspan 向下填充
    assert all(len(r) == 3 for r in grid)


def test_html_to_grid_empty():
    assert table_parser.html_to_grid("") == ([], 0)
    assert table_parser.html_to_grid("<table></table>") == ([], 0)


def test_clean_cell_strips_tags_and_unescapes():
    grid, _ = table_parser.html_to_grid(
        "<table><tr><td><b>1,670</b>&nbsp;万股</td></tr></table>")
    assert grid[0][0] == "1,670 万股"


# ---------------------------------------------------------------------------
# 表头 / 模式识别
# ---------------------------------------------------------------------------
def test_two_col_table_is_kv_mode():
    """2 列表一律按「字段→值」解析，不把首行当列名（工单3 实测修正）。"""
    s = table_parser.parse_table_html(HTML_KV_2COL, "本次发行概况")
    assert s.mode == "kv"
    assert s.header == []
    assert "发行股数" in s.fields
    assert len(s.rows) == 3


def test_definition_table_is_kv_mode():
    """释义表（X | 指 | Y）识别为 kv，值列取"指"右侧那一列。"""
    s = table_parser.parse_table_html(HTML_KV_DEFINITION)
    assert s.mode == "kv"
    assert s.value_cols == [2]
    desc = table_parser.row_description(s, s.rows[1], 1, 5)
    assert "兴图新科有限：武汉兴图新科电子有限公司" in desc
    assert "指：" not in desc                  # 连接词列不进入描述


def test_four_col_table_is_header_mode():
    s = table_parser.parse_table_html(HTML_HEADER_4COL)
    assert s.mode == "header"
    assert s.header[:3] == ["项目", "2019年7-9月", "2018年7-9月"]
    assert len(s.rows) == 2


def test_date_header_not_treated_as_data():
    """日期表头（2010-6-30）不能被误判为数值行（工单3 实测踩坑）。"""
    s = table_parser.parse_table_html(HTML_HEADER_DATE)
    assert s.mode == "header"
    assert s.header == ["项目", "2010-6-30", "2009-12-31"]
    assert len(s.rows) == 2
    assert s.rows[0][0] == "流动资产"


def test_multi_level_header_merged():
    html = ("<table><tr><td>项目</td><td>2019年度</td><td>2018年度</td></tr>"
            "<tr><td></td><td>金额</td><td>金额</td></tr>"
            "<tr><td>营业收入</td><td>1,918.21</td><td>625.12</td></tr></table>")
    s = table_parser.parse_table_html(html)
    assert s.multi_header is True
    assert s.header[0] == "项目"
    assert s.header[1] == "2019年度-金额"


# ---------------------------------------------------------------------------
# 描述文本
# ---------------------------------------------------------------------------
def test_row_description_kv_has_no_duplicate_key():
    s = table_parser.parse_table_html(HTML_KV_2COL, "本次发行概况")
    desc = table_parser.row_description(s, s.rows[1], 1, 3)
    assert desc.startswith("表：本次发行概况 | ")
    assert "发行股数：1,670万股" in desc
    assert desc.endswith("页码：3")
    assert "发行股数 | 发行股数" not in desc


def test_row_description_header_mode_pairs_columns():
    s = table_parser.parse_table_html(HTML_HEADER_4COL)
    desc = table_parser.row_description(s, s.rows[0], 0, 7)
    assert "项目：营业收入" in desc
    assert "2019年7-9月：1,918.21" in desc
    assert "页码：7" in desc


def test_table_heading_text_kv_lists_fields():
    s = table_parser.parse_table_html(HTML_KV_2COL, "本次发行概况")
    head = table_parser.table_heading_text(s, 3)
    assert "表：本次发行概况" in head
    assert "包含字段" in head
    assert "发行股数" in head


def test_row_description_truncates_long_text():
    long_val = "长" * 900
    html = f"<table><tr><td>甲</td><td>{long_val}</td></tr>" \
           f"<tr><td>乙</td><td>短</td></tr></table>"
    s = table_parser.parse_table_html(html)
    desc = table_parser.row_description(s, s.rows[0], 0, 1)
    from src import config
    assert len(desc) <= config.TABLE_ROW_MAX_CHARS


# ---------------------------------------------------------------------------
# 质量分
# ---------------------------------------------------------------------------
def test_quality_high_for_well_formed_table():
    s = table_parser.parse_table_html(HTML_HEADER_4COL)
    assert s.quality >= 0.55
    assert not table_parser.is_low_quality(s)


def test_quality_low_for_empty_table():
    s = table_parser.parse_table_html("")
    assert s.n_cells == 0
    assert table_parser.is_low_quality(s)


def test_quality_low_for_single_cell_table():
    s = table_parser.parse_table_html("<table><tr><td>只有一个格子</td></tr></table>")
    assert table_parser.is_low_quality(s)


# ---------------------------------------------------------------------------
# 表格 chunk
# ---------------------------------------------------------------------------
def _chunks_for(html, caption="测试表", page=4):
    item = {"type": "table", "table_html": html, "text": caption,
            "page_idx": page, "caption": [caption]}
    return table_chunker.chunk_table_item(
        item, source="招股说明书2.pdf", heading_path=["第一节", "释义"],
        table_id="招股说明书2.pdf#p4#t0", chunk_id_start=1, seq_start=0)


def test_chunks_contain_header_and_rows():
    chunks = _chunks_for(HTML_KV_2COL)
    types = [c["block_type"] for c in chunks]
    assert types[0] == "table_header"
    assert types.count("table_row") == 3


def test_chunks_share_parent_table_and_required_metadata():
    """工单硬要求：每个表格 chunk 必须携带 8 项元数据。"""
    chunks = _chunks_for(HTML_KV_2COL)
    for c in chunks:
        assert c["block_type"].startswith("table")
        assert c["page_idx"] == 4
        assert c["source"] == "招股说明书2.pdf"
        assert c["table_id"] == "招股说明书2.pdf#p4#t0"
        assert c["table_html"].startswith("<table")
        assert "table_header" in c
        assert "row_index" in c
        assert c["heading_path"] == ["第一节", "释义"]
        # 父子结构：同一张表的所有 chunk 共享 parent（=整表），供上下文还原
        assert c["parent_id"] == c["table_id"]
    rows = [c for c in chunks if c["block_type"] == "table_row"]
    assert [c["row_index"] for c in rows] == [0, 1, 2]
    assert all(c["parent_text"] for c in rows)


def test_parent_text_is_markdown_table():
    chunks = _chunks_for(HTML_KV_2COL)
    parent = chunks[0]["parent_text"]
    assert "| 项目 | 内容 |" in parent
    assert "| 发行股数 | 1,670万股 |" in parent


def test_chunk_ids_unique_and_sequential():
    chunks = _chunks_for(HTML_HEADER_4COL)
    ids = [c["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids))
    assert ids == sorted(ids)


def test_fallback_when_no_html():
    item = {"type": "table", "table_html": "", "text": "表格纯文本内容",
            "page_idx": 9, "caption": ["某表"]}
    chunks = table_chunker.chunk_table_item(
        item, source="招股说明书1.pdf", heading_path=[],
        table_id="招股说明书1.pdf#p9#t0", chunk_id_start=1, seq_start=0)
    assert len(chunks) == 1
    assert chunks[0]["block_type"] == "table"
    assert chunks[0]["degrade_reason"] == "no_html"


def test_fallback_on_unparsable_html():
    item = {"type": "table", "table_html": "<table><tr></tr></table>",
            "text": "退化文本", "page_idx": 3, "caption": []}
    chunks = table_chunker.chunk_table_item(
        item, source="招股说明书1.pdf", heading_path=[],
        table_id="招股说明书1.pdf#p3#t0", chunk_id_start=1, seq_start=0)
    assert chunks[0]["block_type"] == "table"
    assert chunks[0]["degrade_reason"] == "parse_failed"


def test_ocr_fallback_used_when_html_unparsable():
    item = {"type": "table", "table_html": "<table><tr></tr></table>",
            "text": "", "page_idx": 3, "caption": []}
    chunks = table_chunker.chunk_table_item(
        item, source="招股说明书1.pdf", heading_path=[],
        table_id="招股说明书1.pdf#p3#t0", chunk_id_start=1, seq_start=0,
        ocr_text="OCR 还原出来的表格文字")
    assert len(chunks) == 1
    assert chunks[0]["from_ocr"] is True
    assert "OCR 还原" in chunks[0]["text"]


def test_analyze_tables_flags_low_quality_and_pages():
    items = [
        {"type": "table", "table_html": HTML_HEADER_4COL, "page_idx": 1,
         "caption": []},
        {"type": "table", "table_html": "<table><tr><td>x</td></tr></table>",
         "page_idx": 2, "caption": []},                       # 有 HTML 但低质量
        {"type": "table", "table_html": "", "page_idx": 3, "caption": []},
    ]
    records, low_pages = table_chunker.analyze_tables(items, "招股说明书1.pdf")
    assert len(records) == 3
    # 只有"有 HTML 但解析不出"的才需要 OCR 兜底（无 HTML 的多为版面误判，走文本补齐）
    assert low_pages == [2]


def test_make_table_id_is_unique_per_page():
    a = table_chunker.make_table_id("招股说明书2.pdf", 1, 0)
    b = table_chunker.make_table_id("招股说明书2.pdf", 1, 1)
    c = table_chunker.make_table_id("招股说明书2.pdf", 2, 0)
    assert len({a, b, c}) == 3


def test_table_to_markdown_caption_preserved():
    s = table_parser.parse_table_html(HTML_HEADER_4COL, "经营数据")
    md = table_parser.table_to_markdown(s)
    assert md.splitlines()[0] == "经营数据"


def test_is_usable_requires_rows_and_columns():
    assert not TableStruct().is_usable
    ok = table_parser.parse_table_html(HTML_KV_2COL)
    assert ok.is_usable
