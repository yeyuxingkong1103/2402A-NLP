# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-实现文档质量评估Skill并集成至智能体工作流工单
配置：数据集路径（IMDR documents/ 约 1700 文件）。
"""
import os

ATTACH_DIR = r"D:\软件\QQ\data\RAG 工单1\14-17附件"
DOC_DIR = os.path.join(ATTACH_DIR, "original_problems", "documents")
