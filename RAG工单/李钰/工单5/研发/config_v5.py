# -*- coding: utf-8 -*-
"""
配置模块 V5 - 多轮对话
工单编号: 人工智能 NLP-RAG-Query 理解优化任务
"""
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# LLM 配置 (复用)
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-3.5-turbo")

# 多模态 LLM (复用 V4)
VISION_API_KEY = os.environ.get("VISION_API_KEY", "")

# 会话参数
SESSION_MAX_TURNS = 50     # 最大对话轮次
SESSION_TTL_SECONDS = 3600 # 会话超时 (秒)
MAX_HISTORY_LENGTH = 20    # 保留的历史条数

# PDF 路径 (复用 V3)
PDF_DOCS = [
    {"path": os.path.join(BASE_DIR, "招股说明书1.pdf"), "company": "武汉兴图新科电子股份有限公司"},
    {"path": os.path.join(BASE_DIR, "招股说明书2.pdf"), "company": "武汉力源信息技术股份有限公司"},
]

# 端口
PORT = 5004
