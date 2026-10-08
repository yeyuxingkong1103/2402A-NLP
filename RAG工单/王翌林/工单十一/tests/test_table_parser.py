# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
tests/test_table_parser.py —— 工单三表格解析模块单元测试

覆盖：
  1. table_extractor：单页线框表抽取
  2. table_structurer：表头识别 / 行列 / 合并单元格推断
  3. table_structurer：跨页表格合并（同表头连续 2 页）
  4. table_to_text：表格转自然语言
  5. 容错：无表格页 / 空表不崩溃
  6. 集成：真实附件 PDF（如存在则跑，否则跳过）
"""
import json
from pathlib import Path

import pytest

from src.table_parser.table_extractor import extract_tables_from_pdf, run_full_pipeline
from src.table_parser.table_structurer import structure_tables, _looks_like_header, _row_similarity
from src.table_parser.table_to_text import table_to_text, tables_to_texts

PAGE_W, PAGE_H = 595, 842  # A4


# ================= 辅助：构造测试 PDF =================
def _make_pdf_with_table(path: str, pages_spec, *, draw_lines: bool = True):
    """工单三：构造含表格的测试 PDF

    pages_spec: List[ {table: [[r1c1,r1c2], [r2c1,r2c2]] | None, body: str | None} ]
    """
    import fitz
    doc = fitz.open()
    for spec in pages_spec:
        page = doc.new_page(width=PAGE_W, height=PAGE_H)
        if spec.get("body"):
            page.insert_text(fitz.Point(60, 60), spec["body"], fontsize=11, fontname="china-s")
        tbl = spec.get("table")
        if tbl:
            x0, y0 = 60, 120
            row_h, col_w = 40, 150
            n_rows = len(tbl)
            n_cols = max(len(r) for r in tbl)
            x1 = x0 + n_cols * col_w
            y1 = y0 + n_rows * row_h
            if draw_lines:
                for r in range(n_rows + 1):
                    page.draw_line(fitz.Point(x0, y0 + r * row_h),
                                   fitz.Point(x1, y0 + r * row_h))
                for c in range(n_cols + 1):
                    page.draw_line(fitz.Point(x0 + c * col_w, y0),
                                   fitz.Point(x0 + c * col_w, y1))
            for r, row in enumerate(tbl):
                for c, cell in enumerate(row):
                    page.insert_text(
                        fitz.Point(x0 + 8 + c * col_w, y0 + r * row_h + 25),
                        cell, fontsize=10, fontname="china-s",
                    )
    doc.save(path)
    doc.close()
    return path


@pytest.fixture(scope="module")
def single_table_pdf(tmp_path_factory):
    """单页 + 单张线框表（含表头）"""
    out = tmp_path_factory.mktemp("tbl") / "single.pdf"
    return _make_pdf_with_table(str(out), [
        {
            "body": "测试公司 招股说明书",
            "table": [
                ["项目", "金额（万元）", "比例"],
                ["发行股数", "12 345 678", "100%"],
                ["募集资金总额", "98 765 432", "—"],
            ],
        }
    ])


@pytest.fixture(scope="module")
def crosspage_table_pdf(tmp_path_factory):
    """跨页表格：第 1 页表头 + 2 行，第 2 页重复表头 + 3 行"""
    out = tmp_path_factory.mktemp("tbl") / "cross.pdf"
    return _make_pdf_with_table(str(out), [
        {
            "body": "前五大客户",
            "table": [
                ["客户名称", "金额", "占比"],
                ["客户A", "1 000", "20%"],
                ["客户B", "900", "18%"],
            ],
        },
        {
            "body": "前五大客户（续）",
            "table": [
                ["客户名称", "金额", "占比"],
                ["客户C", "800", "16%"],
                ["客户D", "700", "14%"],
                ["客户E", "600", "12%"],
            ],
        },
    ])


@pytest.fixture(scope="module")
def no_table_pdf(tmp_path_factory):
    """纯文本无表 PDF（容错用）"""
    import fitz
    out = tmp_path_factory.mktemp("tbl") / "notable.pdf"
    doc = fitz.open()
    page = doc.new_page(width=PAGE_W, height=PAGE_H)
    page.insert_text(fitz.Point(60, 60), "这是一段纯文本，没有任何表格。", fontsize=11,
                     fontname="china-s")
    doc.save(str(out))
    doc.close()
    return str(out)


# ================= 1. table_extractor =================
def test_extractor_finds_single_table(single_table_pdf):
    tables = extract_tables_from_pdf(single_table_pdf, doc_id="test_single")
    assert len(tables) == 1, f"应抽出 1 张表，实际 {len(tables)}"
    t = tables[0]
    assert t["doc_id"] == "test_single"
    assert t["page"] == 1
    assert t["source"] == "pdfplumber"
    assert t["col_count"] == 3
    assert t["row_count"] == 3  # 含表头行
    assert t["raw_rows"][0] == ["项目", "金额（万元）", "比例"]


def test_extractor_no_table_pdf_returns_empty(no_table_pdf):
    tables = extract_tables_from_pdf(no_table_pdf, doc_id="test_notable",
                                     use_camelot_fallback=False)
    assert tables == [], f"无表 PDF 应返回空，实际 {len(tables)} 张"


def test_extractor_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        extract_tables_from_pdf(str(tmp_path / "no_such.pdf"))


# ================= 2. table_structurer =================
def test_structurer_basic(single_table_pdf):
    raw = extract_tables_from_pdf(single_table_pdf, doc_id="test_single")
    structured = structure_tables(raw, doc_id="test_single", company="测试公司")
    assert len(structured) == 1
    t = structured[0]
    # 工单三：标准字段
    for k in ("table_id", "page_range", "caption", "headers", "rows",
              "cells", "merged_cells", "header_inferred", "ocr_source"):
        assert k in t, f"缺少字段 {k}"
    assert t["headers"] == ["项目", "金额（万元）", "比例"]
    assert len(t["rows"]) == 2  # 数据行（不含表头）
    assert t["rows"][0][0] == "发行股数"
    assert t["header_inferred"] is False


def test_structurer_inferred_headers_when_no_header(tmp_path_factory):
    """无表头时自动命名 + 标记 header_inferred"""
    out = tmp_path_factory.mktemp("tbl") / "noheader.pdf"
    _make_pdf_with_table(str(out), [
        {"body": "x", "table": [["1 000", "20%"], ["900", "18%"]]}
    ])
    raw = extract_tables_from_pdf(str(out), doc_id="t2", use_camelot_fallback=False)
    structured = structure_tables(raw, doc_id="t2")
    assert len(structured) == 1
    t = structured[0]
    assert t["header_inferred"] is True
    assert t["headers"][0].startswith("col_")
    assert len(t["rows"]) == 2


def test_structurer_crosspage_merge(crosspage_table_pdf):
    """跨页表格应合并为 1 张"""
    raw = extract_tables_from_pdf(crosspage_table_pdf, doc_id="test_cross")
    structured = structure_tables(raw, doc_id="test_cross", company="测试公司")
    assert len(structured) == 1, f"跨页应合并为 1 张，实际 {len(structured)}"
    t = structured[0]
    assert t["page_range"] == [1, 2]
    assert t.get("crosspage_merged") is True
    assert len(t["rows"]) == 5  # 2+3，次页表头被丢弃
    # 表头只保留一次
    assert t["headers"] == ["客户名称", "金额", "占比"]


def test_header_look_like():
    assert _looks_like_header(["项目", "金额", "比例"]) is True
    assert _looks_like_header(["1 000", "20%"]) is False
    assert _looks_like_header(["", ""]) is False


def test_row_similarity():
    a = ["客户名称", "金额", "占比"]
    b = ["客户名称", "金额", "占比"]
    assert _row_similarity(a, b) == 1.0
    c = ["客户A", "1 000", "20%"]
    assert _row_similarity(a, c) < 0.5


# ================= 3. table_to_text =================
def test_table_to_text_basic():
    table = {
        "table_id": "tbl_001",
        "page_range": [3, 3],
        "caption": "发行情况",
        "headers": ["项目", "金额（万元）", "比例"],
        "rows": [
            ["发行股数", "12 345 678", "100%"],
            ["募集资金总额", "98 765 432", "—"],
        ],
        "row_count": 2,
    }
    text = table_to_text(table, doc_id="招股说明书1", company="测试公司")
    assert "表标题: 发行情况" in text
    assert "列: 项目、金额（万元）、比例" in text
    assert "发行股数" in text
    assert "占比 100%" in text
    assert "募集资金总额" in text


def test_tables_to_texts_returns_chunks(single_table_pdf):
    raw = extract_tables_from_pdf(single_table_pdf, doc_id="test_single")
    structured = structure_tables(raw, doc_id="test_single", company="测试公司")
    texts = tables_to_texts(structured, doc_id="test_single", company="测试公司")
    assert len(texts) == 1
    tc = texts[0]
    assert tc["table_chunk_id"].startswith("tc_")
    assert tc["doc_id"] == "test_single"
    assert "测试公司" in tc["table_text"]
    assert "发行股数" in tc["table_text"]


def test_table_to_text_empty_table():
    """空表不崩溃且返回空串"""
    text = table_to_text({"headers": [], "rows": [], "caption": ""})
    assert text == ""


# ================= 4. 全流程 CLI / 集成 =================
def test_run_full_pipeline(single_table_pdf, tmp_path):
    out = tmp_path / "single_tables.json"
    payload = run_full_pipeline(
        single_table_pdf, str(out),
        doc_id="test_single", company="测试公司",
        doc_type="招股说明书", use_camelot_fallback=False,
    )
    assert payload["raw_table_count"] >= 1
    assert payload["structured_table_count"] >= 1
    assert out.exists()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["doc_id"] == "test_single"
    assert data["company"] == "测试公司"
    assert len(data["tables"]) >= 1
    assert len(data["table_texts"]) >= 1


# ================= 5. 集成：真实附件 PDF =================
ATTACH_DIR = Path("/home/dabaie/code/工单/附件")


def _first_attachment_pdf():
    for name in ("招股说明书1.pdf", "招股说明书1-无水印.pdf"):
        p = ATTACH_DIR / name
        if p.exists():
            return str(p)
    return None


@pytest.mark.skipif(_first_attachment_pdf() is None,
                    reason="附件 PDF 不存在，跳过真实集成测试")
def test_real_attachment_pipeline(tmp_path):
    """真实附件 PDF：跑全流程，仅断言不崩溃 + 产出文件 + 至少 1 张表"""
    pdf = _first_attachment_pdf()
    out = tmp_path / "real_tables.json"
    payload = run_full_pipeline(
        pdf, str(out),
        doc_id="招股说明书1",
        company="武汉兴图新科电子股份有限公司",
        doc_type="招股说明书",
        use_camelot_fallback=False,  # camelot 未装，禁用兜底加速
    )
    assert payload["raw_table_count"] >= 1, "真实 PDF 至少抽出 1 张表"
    assert payload["structured_table_count"] >= 1
    assert out.exists()
    # 抽出的表里至少有一张含"发行股数 / 募集资金 / 营业收入 / 客户"关键词之一
    all_text = "\n".join(tc["table_text"] for tc in payload["table_texts"])
    keywords = ["发行股数", "募集资金", "营业收入", "客户", "持股", "比例",
                "金额", "项目"]
    assert any(kw in all_text for kw in keywords), \
        f"抽取的表格应含表格类关键词，实际未命中任何 {keywords}"
