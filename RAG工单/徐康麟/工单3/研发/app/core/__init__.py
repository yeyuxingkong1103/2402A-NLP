# -*- coding: utf-8 -*-
"""工单3 核心业务包：人工智能NLP-RAG-PDF文档的表格解析及检索优化。

模块清单（按设计 §3 冻结）：
    config        配置与语料发现
    logging_conf  自实现结构化 JSON 日志（禁 loguru）
    errors        统一异常体系
    text_utils    文本/数字/分词/命中判定工具
    table_parser  表格归一化与跨页续表合并
    pdf_parser    PDF 逐页解析与产物落盘
    chunker       分块、关键词与元数据
"""
