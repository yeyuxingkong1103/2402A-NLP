# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG项目-RAG性能瓶颈识别与优化
配置：PDF 路径、目标延迟、LLM 配置、优化开关。
"""
import os

ATTACH_DIR = r"D:\软件\QQ\data\RAG 工单\附件"
PDF1 = os.path.join(ATTACH_DIR, "招股说明书1.pdf")
PDF2 = os.path.join(ATTACH_DIR, "招股说明书2.pdf")

# 验收标准：每个会话返回检索结果要在 3 秒以内
TARGET_LATENCY = 3.0

# 离线模式模拟 LLM 推理延迟（秒），配置真实 API Key 时走真实接口
SIMULATED_LLM_LATENCY = 0.3

LLM_API_KEY = os.environ.get("OPENAI_API_KEY") or os.environ.get("DASHSCOPE_API_KEY")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")

TOP_K = 5

# 测试查询（用于优化前后对比）
TEST_QUERIES = [
    "武汉力源信息技术股份有限公司本次发行股数是多少？",
    "武汉兴图新科电子股份有限公司法定代表人是谁？",
    "报告期内武汉兴图新科电子股份有限公司来自军用领域的收入是多少？",
    "武汉力源信息技术股份有限公司本次募集资金拟投资哪些项目？",
    "武汉兴图新科电子股份有限公司注册资本是多少？",
]
