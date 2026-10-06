# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化
tests/test_pdf_parser_optimized.py —— 工单二优化版 PDF 解析器单元测试

覆盖：
  1. 输出 JSON 必填字段完整性（layout_blocks / tables_structured / headings）
  2. 版面分析：标题识别与层级、页眉页脚标记
  3. 表格结构化：表头/行/Markdown
  4. OCR 兜底：引擎不可用时优雅降级（ocr_skipped 标记，不崩溃）
"""
import pytest

from src.pdf_parser_optimized import parse_pdf_optimized, _table_to_markdown

PAGE_W, PAGE_H = 595, 842  # A4


@pytest.fixture(scope="module")
def sample_pdf(tmp_path_factory):
    """构造测试 PDF：6 页（满足页眉频次阈值），含标题/正文/页眉/线框表格/纯图空页"""
    import fitz
    out = tmp_path_factory.mktemp("pdfs") / "sample_opt.pdf"
    doc = fitz.open()
    for i in range(6):
        page = doc.new_page(width=PAGE_W, height=PAGE_H)
        # 页眉（每页相同，覆盖 100% ≥ 50% 阈值；china-s 字体支持中文）
        page.insert_text(fitz.Point(60, 25), f"测试公司 招股说明书 第{i + 1}页", fontsize=9,
                         fontname="china-s")
        if i == 4:
            # 第 5 页：线框表格（2 列 × 3 行）
            x0, y0, x1, y1 = 60, 100, 360, 220
            row_h, col_w = 40, 150
            for r in range(4):
                page.draw_line(fitz.Point(x0, y0 + r * row_h), fitz.Point(x1, y0 + r * row_h))
            for c in range(3):
                page.draw_line(fitz.Point(x0 + c * col_w, y0), fitz.Point(x0 + c * col_w, y1))
            cells = [("项目", "金额"), ("营业收入", "100万"), ("净利润", "20万")]
            for r, (c1, c2) in enumerate(cells):
                page.insert_text(fitz.Point(x0 + 8, y0 + r * row_h + 25), c1, fontsize=10,
                                 fontname="china-s")
                page.insert_text(fitz.Point(x0 + col_w + 8, y0 + r * row_h + 25), c2, fontsize=10,
                                 fontname="china-s")
        elif i == 5:
            # 第 6 页：纯图空页（触发 OCR 兜底路径）——插入小图占位但无文字
            page.draw_rect(fitz.Rect(50, 50, 300, 300), color=(0, 0, 0), fill=None)
        else:
            # 标题（大字号 20pt）+ 正文（10pt）
            page.insert_text(fitz.Point(60, 120), "第三章 业务和技术", fontsize=20,
                             fontname="china-s")
            page.insert_text(fitz.Point(60, 160), "公司主要从事电子信息系统的研发与销售。",
                             fontsize=10, fontname="china-s")
    doc.save(str(out))
    doc.close()
    return str(out)


@pytest.fixture(scope="module")
def parsed(sample_pdf):
    return parse_pdf_optimized(sample_pdf, enable_ocr=True)


# ---------- 1. 字段完整性（人工智能NLP-RAG-基于PDF文档的问答系统优化） ----------
def test_required_fields(parsed):
    for key in ("doc_id", "filename", "total_pages", "pages", "text",
                "layout_blocks", "tables_structured", "headings", "metadata", "errors"):
        assert key in parsed, f"缺少字段: {key}"
    assert parsed["total_pages"] == 6
    assert parsed["doc_id"]


def test_no_errors(parsed):
    assert parsed["errors"] == [], f"解析异常: {parsed['errors']}"


# ---------- 2. 版面分析（人工智能NLP-RAG-基于PDF文档的问答系统优化） ----------
def test_headings_detected(parsed):
    hs = [h for h in parsed["headings"] if "第三章" in h["text"]]
    assert len(hs) == 4, f"标题应出现 4 次（第1~4页），实际 {len(hs)}: {parsed['headings'][:5]}"
    assert all(h["level"] == 1 for h in hs)
    assert all(h["font_size"] >= 12 for h in hs)


def test_layout_blocks_types(parsed):
    types = {b["type"] for b in parsed["layout_blocks"]}
    assert {"heading", "body"} <= types, f"布局类型缺失: {types}"


def test_header_footer_marked(parsed):
    hf = [b for b in parsed["layout_blocks"] if b["type"] in ("header", "footer")]
    assert len(hf) >= 6, f"页眉应至少标记 6 处，实际 {len(hf)}"
    assert all("招股说明书" in b["text"] for b in hf)


# ---------- 3. 表格结构化（人工智能NLP-RAG-基于PDF文档的问答系统优化） ----------
def test_table_structured(parsed):
    ts = parsed["tables_structured"]
    assert len(ts) == 1, f"应提取 1 张结构化表格，实际 {len(ts)}"
    t = ts[0]
    assert t["page"] == 5 and t["engine"] == "pdfplumber"
    assert len(t["headers"]) == 2 and len(t["rows"]) >= 2
    assert t["markdown"].startswith("| 项目 | 金额 |")


def test_table_to_markdown_helper():
    md = _table_to_markdown(["A", "B"], [["1", "2"], ["3", "4"]])
    assert "| A | B |" in md and "| --- | --- |" in md and "| 3 | 4 |" in md


def test_legacy_tables_compat(parsed):
    """兼容工单一格式：tables 字段保留原始行列"""
    assert len(parsed["tables"]) == 1
    assert parsed["tables"][0]["page"] == 5


# ---------- 4. OCR 兜底优雅降级（人工智能NLP-RAG-基于PDF文档的问答系统优化） ----------
def test_ocr_graceful_degradation(parsed):
    """第 6 页无文字：OCR 引擎不可用时标记 ocr_skipped 且不崩溃"""
    last = parsed["pages"][-1]
    assert last["page"] == 6
    # 测试环境无 tesseract 二进制 → 必须被标记（若环境恰好可用则 OCR 成功也放行）
    if not last.get("ocr"):
        assert last.get("ocr_skipped") is True


def test_body_text_extracted(parsed):
    assert "电子信息系统的研发与销售" in parsed["text"]
