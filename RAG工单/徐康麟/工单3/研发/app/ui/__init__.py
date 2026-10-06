# -*- coding: utf-8 -*-
"""工单3 界面包（设计/接口设计.md §3.25）。

工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化

* ``streamlit_app``：真正的 Streamlit 应用（算力云可跑，模块顶层 ``import streamlit``）；
* ``serve_fallback``：纯标准库 ``http.server`` 备用界面（本机演示可用）。

两者**只**通过 ``app.core.qa_engine`` 访问业务逻辑，不得各自实现检索/生成/引用。
"""

from __future__ import annotations

__all__ = ["streamlit_app", "serve_fallback"]
