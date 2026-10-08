# -*- coding: utf-8 -*-
"""功能2：PDF 文字型/扫描型/混合型分流（逐页启发式）。"""
from __future__ import annotations

from pathlib import Path

from .text_reader import open_pdf_pages


def classify_pdf(path: Path, config: dict, pages_info=None) -> dict:
    """按页判断 PDF 类型。

    pages_info: 可选 (page_count, page_chars)，提供则不重复打开文件。
    返回: {type: Text_PDF/Scan_PDF/Hybrid_PDF/Corrupt_PDF,
           page_count, scanned_pages, scanned_ratio, near_boundary(是否临界待确认)}
    """
    char_thr = config["text_char_threshold"]
    scan_ratio_thr = config["scanned_file_ratio"]
    mixed_min = config["mixed_min_ratio"]

    if pages_info is None:
        try:
            page_count, page_chars = open_pdf_pages(path)
        except Exception as e:
            return {"type": "Corrupt_PDF", "page_count": 0, "scanned_pages": 0,
                    "scanned_ratio": 0.0, "near_boundary": False, "error": str(e)}
    else:
        page_count, page_chars = pages_info

    if page_count == 0:
        return {"type": "Corrupt_PDF", "page_count": 0, "scanned_pages": 0,
                "scanned_ratio": 0.0, "near_boundary": False, "error": "0 页"}

    scanned_pages = sum(1 for c in page_chars if c < char_thr)
    ratio = scanned_pages / page_count

    if ratio > scan_ratio_thr:
        pdf_type = "Scan_PDF"
    elif ratio > mixed_min:
        pdf_type = "Hybrid_PDF"
    else:
        pdf_type = "Text_PDF"

    # 临界样本（扫描页占比 60%~80%）进入待确认列表
    near_boundary = 0.6 <= ratio <= 0.8

    return {
        "type": pdf_type,
        "page_count": page_count,
        "scanned_pages": scanned_pages,
        "scanned_ratio": round(ratio, 4),
        "near_boundary": near_boundary,
        "text_pages": page_count - scanned_pages,
    }
