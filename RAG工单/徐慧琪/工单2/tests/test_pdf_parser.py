# -*- coding: utf-8 -*-
"""工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化 —— 解析模块测试"""
import json
import os

import pytest

from src import config


def _fake_content_list(tmp_path):
    items = [
        {"type": "text", "text": "第一节 释义", "page_idx": 0, "text_level": 1},
        {"type": "text", "text": "本公司指武汉兴图新科电子股份有限公司。" * 5, "page_idx": 0},
        {"type": "table", "table_body": "<table><tr><td>项目</td><td>金额</td></tr></table>",
         "page_idx": 1, "table_caption": ["表1 收入"]},
        {"type": "equation", "text": "E = mc^2", "text_format": "latex", "page_idx": 2},
        {"type": "image", "img_path": "images/a.jpg", "page_idx": 2, "image_caption": ["图1"]},
        {"type": "unknown-type", "text": "忽略我", "page_idx": 3},
    ]
    p = os.path.join(str(tmp_path), "content_list.json")
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(items, fh, ensure_ascii=False)
    return p


def test_load_content_list_normalizes(tmp_path):
    from src.pdf_parser import load_content_list
    items = load_content_list(_fake_content_list(tmp_path))
    assert len(items) == 5                       # 未知类型被过滤
    assert items[0]["type"] == "text" and items[0]["text_level"] == 1
    # MinerU 的 HTML 表格会被转为紧凑 Markdown（见 _html_table_to_markdown）
    assert items[2]["type"] == "table" and "| 项目 | 金额 |" in items[2]["text"]
    assert items[2]["caption"] == ["表1 收入"]
    assert items[3]["type"] == "equation"
    assert items[4]["type"] == "image" and items[4]["text"] == "图1"


def test_quality_report_flags_low_text_pages(tmp_path):
    from src.pdf_parser import load_content_list, quality_report
    items = load_content_list(_fake_content_list(tmp_path))
    rep = quality_report(items, page_count=5)
    assert rep["n_tables"] == 1 and rep["n_equations"] == 1 and rep["n_images"] == 1
    assert 3 in rep["low_text_pages"] and 4 in rep["low_text_pages"]
    assert 0 not in rep["low_text_pages"]


@pytest.mark.slow
def test_parse_pdf_fallback_pymupdf(tmp_path):
    """force_fallback 时必须走 PyMuPDF，结构完整、页数正确。"""
    from src import pdf_parser
    out = pdf_parser.parse_pdf(config.SOURCE_PDF, str(tmp_path), reuse=False,
                               force_fallback=True)
    assert out["parser"] == "pymupdf"
    assert out["page_count"] == 548
    assert len(out["items"]) > 100
    assert all("page_idx" in it and "type" in it for it in out["items"])
