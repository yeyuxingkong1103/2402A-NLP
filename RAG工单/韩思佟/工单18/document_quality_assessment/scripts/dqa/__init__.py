# -*- coding: utf-8 -*-
"""DQA —— Document Quality Assessment（文档质量评估）
人工智能NLP-RAG-实现文档质量评估 Skill 并集成至智能体工作流工单
"""
__version__ = "1.0.0"

from .format_stats import analyze_format_distribution
from .pdf_type import classify_pdf_pages
from .length_dist import analyze_length_distribution
from .dedup import detect_duplicates
from .sensitive import detect_sensitive_info
from .classifier import classify_documents
from .report import build_report, to_html, to_markdown

__all__ = [
    "analyze_format_distribution", "classify_pdf_pages",
    "analyze_length_distribution", "detect_duplicates",
    "detect_sensitive_info", "classify_documents",
    "build_report", "to_html", "to_markdown",
]
