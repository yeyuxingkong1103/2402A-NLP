# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-LightRAG优化
"""
全局配置模块。

工单：人工智能NLP-RAG项目-LightRAG优化任务
内容：LightRAG 知识图谱构建 / 传统 RAG 与 LightRAG 检索对比 / RAGAS 评估
"""

import os
from pathlib import Path

# ---------------- 路径配置 ----------------
SRC_DIR = Path(__file__).resolve().parent
DATA_DIR = SRC_DIR / "data"                       # 原始 PDF
WORK_DIR = SRC_DIR / "lightrag_data"              # LightRAG 工作目录（图谱存储）
TXT_DIR = SRC_DIR / "extracted"                   # PDF 提取文本
RESULT_DIR = SRC_DIR / "results"                  # 运行结果输出
LOG_DIR = SRC_DIR / "logs"                        # 日志目录
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "lightrag_rag.log"           # 日志文件
LOG_LEVEL = "INFO"                                # 日志级别

for _d in (DATA_DIR, WORK_DIR, TXT_DIR, RESULT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# 两份招股说明书
PDF_FILES = {
    "招股说明书1.pdf": DATA_DIR / "招股说明书1.pdf",
    "招股说明书2.pdf": DATA_DIR / "招股说明书2.pdf",
}

# ---------------- LLM 配置（DeepSeek API） ----------------
LLM_API_KEY = os.getenv("deepseek_api_key1") or os.getenv("DEEPSEEK_API_KEY1")
LLM_BASE_URL = (
    os.getenv("deepseek_base_url")
    or os.getenv("DEEPSEEK_BASE_URL")
    or os.getenv("deepseek_base_url1")
    or "https://api.deepseek.com"
).rstrip("/")
LLM_MODEL = "deepseek-chat"
LLM_TEMPERATURE = 0.0
LLM_MAX_TOKENS = 4096

# ---------------- Embedding 配置 ----------------
# 本地 bge-m3 模型（中文语义向量，1024 维）
EMBED_MODEL_PATH = r"D:\Projects\Models\bge-m3"
EMBED_DIM = 1024
EMBED_BATCH_SIZE = 16

# ---------------- LightRAG 配置 ----------------
CHUNK_TOKEN_SIZE = 1024          # 文本切块 token 上限
CHUNK_OVERLAP_TOKEN = 100        # 切块重叠 token
LLM_MAX_ASYNC = 16               # LLM 并发数
EMBED_MAX_ASYNC = 4              # Embedding 并发数
EMBED_BATCH_NUM = 16             # Embedding 批大小
EMBED_TIMEOUT = 600              # Embedding 超时（秒）——CPU 批量编码 chunk 较慢，默认 30s 会超时

# ---------------- 检索配置 ----------------
TOP_K = 10                       # 检索返回条数
CHUNK_TOP_K = 8                  # 参与上下文组装的 chunk 数

# ---------------- 测试问题集 ----------------
QUESTIONS_FILE = SRC_DIR / "questions.py"
