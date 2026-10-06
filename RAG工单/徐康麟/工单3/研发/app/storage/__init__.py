# -*- coding: utf-8 -*-
"""工单3 存储层包（设计/接口设计.md §3.21）。

工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
"""

from __future__ import annotations

from .sqlite_manager import SQLiteManager, get_sqlite_manager

__all__ = ["SQLiteManager", "get_sqlite_manager"]
