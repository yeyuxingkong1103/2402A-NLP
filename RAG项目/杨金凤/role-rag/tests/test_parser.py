"""parser.py 单元测试：extract_text / extract_tables / 去水印 / 表格 Markdown / 图片。"""
from pathlib import Path

import pytest

import parser


def _make_pdf_bytes() -> bytes:
    """构造合法 3 页最小 PDF：第 1/3 页含 ASCII 文本，第 2 页为空页。"""
    stream1 = "BT /F1 24 Tf 72 720 Td (Hello World) Tj ET"
    stream3 = "BT /F1 24 Tf 72 720 Td (Page Three) Tj ET"

    objects = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [3 0 R 4 0 R 5 0 R] /Count 3 >>",
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        "/Resources << /Font << /F1 6 0 R >> >> /Contents 7 0 R >>",
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>",
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        "/Resources << /Font << /F1 6 0 R >> >> /Contents 8 0 R >>",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        f"<< /Length {len(stream1)} >>\nstream\n{stream1}\nendstream",
        f"<< /Length {len(stream3)} >>\nstream\n{stream3}\nendstream",
    ]

    out = bytearray(b"%PDF-1.4\n")
    offsets = [0]  # offsets[i] = 对象 i 的起始字节偏移
    for i, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode()
        out += body.encode()
        out += b"\nendobj\n"

    xref_pos = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for i in range(1, len(objects) + 1):
        out += f"{offsets[i]:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_pos}\n%%EOF\n"
    ).encode()
    return bytes(out)


# ---- extract_text ----

def test_extract_text_skips_empty_and_numbers_from_one(tmp_path):
    """3 页（文本/空/文本）：页码从 1 起、空页跳过、页码不重排。"""
    p = tmp_path / "t.pdf"
    p.write_bytes(_make_pdf_bytes())
    pages = parser.extract_text(str(p))
    assert [pg["page"] for pg in pages] == [1, 3]


def test_extract_text_returns_trimmed_text(tmp_path):
    """抽出的文本与页序一致，且已去除首尾空白。"""
    p = tmp_path / "t.pdf"
    p.write_bytes(_make_pdf_bytes())
    pages = parser.extract_text(str(p))
    assert [pg["text"] for pg in pages] == ["Hello World", "Page Three"]
    assert all(pg["text"] == pg["text"].strip() for pg in pages)


# ---- extract_tables ----

def test_extract_tables_no_tables_returns_empty(tmp_path):
    p = tmp_path / "t.pdf"
    p.write_bytes(_make_pdf_bytes())
    assert parser.extract_tables(str(p)) == []


# ---- _render_table_markdown ----

def test_render_table_markdown_merges_multiline_header():
    table = [["服药\n（次/天）", "主要\n不良反应2"], ["1~2", "咳嗽"]]
    assert parser._render_table_markdown(table) == (
        "| 服药（次/天） | 主要不良反应2 |\n"
        "| --- | --- |\n"
        "| 1~2 | 咳嗽 |"
    )


def test_render_table_markdown_skips_empty_table():
    assert parser._render_table_markdown([[""], [""], [""]]) is None


def test_render_table_markdown_skips_single_row_table():
    assert parser._render_table_markdown([["高血压合并疾病/情况", "LDL-C目标值"]]) is None


# ---- _remove_watermark ----

def test_remove_watermark_removes_repeated_short_lines():
    pages = [
        [(20, "···· ····"), (100, "正文甲")],
        [(20, "···· ····"), (100, "正文乙")],
        [(20, "···· ····"), (100, "正文丙")],
    ]
    cleaned, removed = parser._remove_watermark(pages)
    assert removed == 3
    assert cleaned == [[(100, "正文甲")], [(100, "正文乙")], [(100, "正文丙")]]


def test_remove_watermark_keeps_short_line_on_fewer_than_three_pages():
    pages = [[(20, "页脚")], [(20, "页脚")], [(100, "正文")]]
    cleaned, removed = parser._remove_watermark(pages)
    assert removed == 0
    assert cleaned == pages


def test_remove_watermark_keeps_long_repeated_text():
    long_text = "很长的文本" * 10  # 50 字
    pages = [[(20, long_text)], [(20, long_text)], [(20, long_text)]]
    cleaned, removed = parser._remove_watermark(pages)
    assert removed == 0
    assert cleaned == pages


# ---- extract_images ----

def test_extract_images_no_images_returns_empty(tmp_path):
    p = tmp_path / "t.pdf"
    p.write_bytes(_make_pdf_bytes())
    assert parser.extract_images(str(p)) == []


# ---- _filter_header_footer ----

def test_filter_header_footer_removes_repeated_lines():
    pages = [
        ["医院指南", "第 1 章", "正文甲", "页脚1"],
        ["医院指南", "第 2 章", "正文乙", "页脚1"],
    ]
    cleaned = parser._filter_header_footer(pages, header_size=2, footer_size=1)
    assert cleaned == [
        ["第 1 章", "正文甲"],
        ["第 2 章", "正文乙"],
    ]


def test_filter_header_footer_keeps_unique_lines():
    pages = [["甲", "正文1"], ["乙", "正文2"]]
    cleaned = parser._filter_header_footer(pages, header_size=1, footer_size=0)
    assert cleaned == pages


@pytest.mark.slow
def test_extract_from_image_ocr_returns_text():
    """OCR 识别真实图片（需 EasyOCR + 素材图，标 slow）。"""
    pytest.importorskip("easyocr")
    img = Path(__file__).resolve().parent.parent / "data" / "raw" / "images" / "guide_p5.png"
    if not img.exists():
        pytest.skip("缺少测试图片 data/raw/images/guide_p5.png")
    pages = parser.extract_from_image(str(img))
    assert pages and pages[0]["text"].strip()
