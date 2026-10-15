# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-18-实现文档质量评估Skill并集成至智能体工作流
# 模块职责：document_ingestion_workflow 包入口，暴露工作流类与技能注册函数。

from .workflow import DocumentIngestionWorkflow, register_skill, WORKFLOW_NAME

__all__ = ["DocumentIngestionWorkflow", "register_skill", "WORKFLOW_NAME"]
