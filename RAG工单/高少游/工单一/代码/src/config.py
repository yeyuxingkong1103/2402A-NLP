# -*- coding: utf-8 -*-
"""项目全局配置
工单编号: 人工智能 NLP-RAG-基于 PDF 文档的问答系统

说明：
系统基于本地 langchain2 环境（Python 3.10 + LangChain 1.3 + Ollama 本地模型）运行，
无需外部 API Key。所有可调参数优先读取环境变量 / .env，其次使用默认值。
"""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ---- 目录与路径 -------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent    # 项目根目录
DATA_DIR = BASE_DIR / "data"                         # 数据目录（PDF、题目）
DB_DIR = BASE_DIR / "vector_db"                      # Chroma 向量数据库目录
OUTPUT_DIR = BASE_DIR / "output"                     # 评估结果、日志输出目录

PDF_PATH = DATA_DIR / "招股说明书1.pdf"               # 知识库源文档
QUESTIONS_PATH = DATA_DIR / "questions.json"         # 待评估问题清单

# ---- 模型配置（Ollama 本地模型） ---------------------------------------------
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-r1:1.5b")     # 本地大模型（生成）
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "bge-m3:567m")  # 本地向量模型

LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "768"))    # 控制生成长度以优化响应时间
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "60"))

# ---- 检索与切片配置 ----------------------------------------------------------
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))            # 文本切片大小
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "80"))       # 切片重叠
TOP_K = int(os.getenv("TOP_K", "6"))                        # 向量检索返回片段数
BM25_TOP_K = int(os.getenv("BM25_TOP_K", "6"))              # BM25 检索返回片段数
RANK_FUSION_K = int(os.getenv("RANK_FUSION_K", "60"))       # 倒数排名融合窗口

VECTOR_COLLECTION = os.getenv("VECTOR_COLLECTION", "prospectus_qa")

# ---- 性能目标 -----------------------------------------------------------------
TARGET_RESPONSE_SECONDS = 3.0   # 需求：从提问到生成答案 <= 3s


def ensure_dirs() -> None:
    """确保关键目录存在（首次运行自动创建）。"""
    for d in (DATA_DIR, DB_DIR, OUTPUT_DIR):
        try:
            d.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass


ensure_dirs()