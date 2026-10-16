# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-Graph RAG 优化任务
# 说明：本文件复用自工单8（工单9 未修改其业务逻辑），仅补齐工单9 的编号头注释。
"""工单9 界面包：Web 界面（纯标准库兜底 + 真 Streamlit 两个实现）。

工单编号：人工智能NLP-RAG-Graph RAG 优化任务

纪律（设计/接口设计.md §3.6）：``serve_fallback.py`` 与 ``streamlit_app.py``
**必须共用同一 ``app.core`` 业务逻辑**（同一个 ``QAEngine``），不得各写一套检索/生成。
"""

from .serve_fallback import DEMO_QUESTIONS, TITLE, build_server, serve

__all__ = ["serve", "build_server", "DEMO_QUESTIONS", "TITLE"]
