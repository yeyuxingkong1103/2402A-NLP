"""离线测试：PDF 解析（``app/core/pdf_parser.py``）。

测试目标（工单 9.1 / 验收标准 1）：
1. 解析依赖可用（PyMuPDF 必须可用，pdfplumber 为可选的表格备选解析器）；
2. 全文页数与 PDF 物理页数一致（语料 548 页）、页码从 1 开始且连续；
3. 能提取正文文字，且页眉页脚/孤立页码等噪声被清洗；
4. 能提取表格，表格带 ``table_id`` / ``page`` / ``markdown`` 且页码落在文档范围内；
5. 解析结果可落盘并原样回读；
6. 异常输入（文件不存在）必须显式报错——禁止静默失败。

加速策略：除"页数校验"外的用例都用 ``max_pages`` 小值解析，避免整本 548 页重复解析。
"""

from __future__ import annotations

import re

import pytest

from app.core.pdf_parser import PDFParser, parser_capabilities

# 语料固定页数：工单明确要求验证
EXPECTED_PAGE_COUNT = 548
# 表格 ID 形如 p152_t1
TABLE_ID_PATTERN = re.compile(r"^p(\d+)_t(\d+)$")


# --------------------------------------------------------------------------
# 能力与环境
# --------------------------------------------------------------------------
def test_parser_capabilities_pymupdf_available() -> None:
    """PyMuPDF 是主解析器，必须可用，否则整条链路无法建立索引。"""
    capabilities = parser_capabilities()
    assert capabilities["pymupdf"] is True, "未检测到 PyMuPDF，PDF 正文解析不可用"


# --------------------------------------------------------------------------
# 页数与页码
# --------------------------------------------------------------------------
def test_full_document_page_count(settings, pdf_page_count) -> None:
    """全文解析的页数必须等于 PDF 物理页数，且为语料约定的 548 页。"""
    document = PDFParser().parse(settings.paths.default_pdf, max_pages=0, extract_tables=False)
    assert document.page_count == pdf_page_count, (
        f"解析页数({document.page_count})与 PDF 物理页数({pdf_page_count})不一致"
    )
    assert document.page_count == EXPECTED_PAGE_COUNT, (
        f"语料《招股说明书1.pdf》应为 {EXPECTED_PAGE_COUNT} 页，实际解析到 {document.page_count} 页"
    )
    assert len(document.pages) == document.page_count, "pages 列表长度必须与 page_count 一致"


def test_page_numbers_start_at_one_and_are_continuous(settings) -> None:
    """页码必须从 1 开始、严格连续，引用页码才能与 PDF 阅读器对应。"""
    document = PDFParser().parse(settings.paths.default_pdf, max_pages=20, extract_tables=False)
    numbers = [page.page for page in document.pages]
    assert numbers == list(range(1, 21)), f"页码应为 1..20 的连续序列，实际为 {numbers[:10]}..."


# --------------------------------------------------------------------------
# 正文
# --------------------------------------------------------------------------
def test_text_extraction_produces_content(settings) -> None:
    """前 20 页必须能提取到正文，且绝大多数页有实质内容。"""
    document = PDFParser().parse(settings.paths.default_pdf, max_pages=20, extract_tables=False)
    total_chars = sum(page.char_count for page in document.pages)
    assert total_chars > 2000, f"前 20 页正文总字数仅 {total_chars}，文本层可能缺失（扫描件需 OCR）"

    non_empty = [page for page in document.pages if len(page.text.strip()) >= 20]
    assert len(non_empty) >= 18, (
        f"前 20 页中仅 {len(non_empty)} 页提取到有效正文（>=20 字），文本提取质量不足"
    )
    for page in document.pages:
        assert page.char_count == len(page.text), f"第 {page.page} 页 char_count 与 text 长度不一致"


def test_clean_page_text_removes_header_footer() -> None:
    """页眉（公司名+招股意向书）、页脚（1-1-35 之类）与孤立短行必须被清洗。"""
    raw = (
        "武汉兴图新科电子股份有限公司 招股意向书\n"
        "1-1-35\n"
        "第五节 业务与技术\n"
        "A\n"
        "公司是国内领先的军用视频指挥系统供应商。\n"
    )
    cleaned = PDFParser().clean_page_text(raw)
    assert "招股意向书" not in cleaned, "页眉（公司名 + 招股意向书）未被清洗"
    assert "1-1-35" not in cleaned, "页脚形式的页码（1-1-35）未被清洗"
    assert "第五节 业务与技术" in cleaned, "正文标题被误删"
    assert "公司是国内领先的军用视频指挥系统供应商。" in cleaned, "正文内容被误删"


# --------------------------------------------------------------------------
# 表格
# --------------------------------------------------------------------------
def test_tables_are_extracted_with_metadata(sample_document) -> None:
    """前 30 页必须能提取到表格，且每张表都带 table_id / page / markdown。"""
    tables = sample_document.tables
    assert tables, "前 30 页未提取到任何表格，招股书关键财务数据将无法检索"

    for table in tables:
        match = TABLE_ID_PATTERN.match(table.table_id)
        assert match, f"表格 ID 格式应为 p<页码>_t<序号>，实际为 {table.table_id!r}"
        assert int(match.group(1)) == table.page, (
            f"表格 {table.table_id} 的页码字段({table.page})与 ID 中的页码({match.group(1)})不一致"
        )
        assert 1 <= table.page <= sample_document.page_count, (
            f"表格 {table.table_id} 的页码 {table.page} 超出文档范围 1..{sample_document.page_count}"
        )
        assert table.markdown.strip(), f"表格 {table.table_id} 的 markdown 为空"
        assert "|" in table.markdown, f"表格 {table.table_id} 未转成 Markdown 形式"
        assert table.rows, f"表格 {table.table_id} 缺少二维原始单元格数据"


def test_table_markdown_has_header_separator(sample_document) -> None:
    """Markdown 表格必须含分隔行（``| --- |``），否则 LLM 无法识别表头。"""
    tables = [table for table in sample_document.tables if table.markdown]
    assert tables, "没有可用于校验的表格"
    sample = tables[0]
    lines = sample.markdown.splitlines()
    assert len(lines) >= 2, f"表格 {sample.table_id} 的 Markdown 少于 2 行，缺少表头或分隔行"
    assert set(lines[1].replace("|", " ").split()) <= {"---"}, (
        f"表格 {sample.table_id} 的第 2 行应为 Markdown 分隔行，实际为：{lines[1]!r}"
    )


def test_table_page_limit_respected(settings) -> None:
    """``max_pages`` 必须真正限制解析范围（加速用例的正确性保障）。"""
    document = PDFParser().parse(settings.paths.default_pdf, max_pages=5, extract_tables=True)
    assert document.page_count == 5, f"max_pages=5 时应只解析 5 页，实际 {document.page_count} 页"
    assert all(table.page <= 5 for table in document.tables), (
        "限制页数后仍解析出了范围外的表格：" + str([table.page for table in document.tables])
    )


# --------------------------------------------------------------------------
# 落盘与异常
# --------------------------------------------------------------------------
def test_save_and_load_roundtrip(sample_document, tmp_path) -> None:
    """解析结果落盘后必须能原样回读（避免重复解析整本 PDF）。"""
    parser = PDFParser()
    paths = parser.save(sample_document, output_dir=tmp_path)

    for key in ("parsed_json", "pages_txt", "tables_jsonl"):
        assert paths[key].exists(), f"解析产物 {key} 未生成：{paths[key]}"
        assert paths[key].stat().st_size > 0, f"解析产物 {key} 为空文件：{paths[key]}"

    restored = PDFParser.load_parsed(paths["parsed_json"])
    assert restored.doc_id == sample_document.doc_id, "回读后的 doc_id 与原文档不一致"
    assert restored.page_count == sample_document.page_count, "回读后的页数与原文档不一致"
    assert len(restored.pages) == len(sample_document.pages), "回读后的页对象数量与原文档不一致"
    assert len(restored.tables) == len(sample_document.tables), "回读后的表格数量与原文档不一致"
    assert restored.pages[0].text == sample_document.pages[0].text, "回读后的正文内容与原文档不一致"


def test_missing_file_raises_file_not_found(tmp_path) -> None:
    """传入不存在的 PDF 必须显式抛错，禁止静默返回空文档。"""
    with pytest.raises(FileNotFoundError, match="PDF 文件不存在"):
        PDFParser().parse(tmp_path / "不存在的文件.pdf")
