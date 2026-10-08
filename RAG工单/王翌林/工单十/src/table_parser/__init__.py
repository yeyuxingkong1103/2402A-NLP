# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
src/table_parser/__init__.py —— 工单三表格解析模块包入口

提供 PDF 表格结构化解析能力：
  - table_extractor: 表格区域定位与原始抽取（pdfplumber + camelot 兜底）
  - table_structurer: 结构化（合并单元格 / 表头 / 跨页合并）
  - table_to_text: 表格转自然语言描述（用于向量化）
"""

from .table_extractor import extract_tables_from_pdf, run_full_pipeline  # noqa: F401
from .table_structurer import structure_tables  # noqa: F401
from .table_to_text import table_to_text, tables_to_texts  # noqa: F401

__all__ = [
    "extract_tables_from_pdf",
    "run_full_pipeline",
    "structure_tables",
    "table_to_text",
    "tables_to_texts",
]
__version__ = "1.0.0"
