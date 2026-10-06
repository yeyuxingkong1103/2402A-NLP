# -*- coding: utf-8 -*-
"""项目全局配置（优化版）
工单编号: 人工智能 NLP-RAG-基于 PDF 文档的问答系统优化

说明：
本工单在《01-基于 PDF 文档的问答系统》基础上，围绕
「PDF 解析处理 / 分块优化 / 检索优化」三个层面做检索准确率优化。
所有可调参数优先读取环境变量 / .env，其次使用默认值。
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
DB_DIR = BASE_DIR / "vector_db" / "optimized"        # 【优化后】FAISS 向量库目录
BASE_DB_DIR = BASE_DIR / "vector_db" / "baseline"    # 【优化前】FAISS 向量库目录
OUTPUT_DIR = BASE_DIR / "output"                     # 评估结果、日志输出目录
FIGURE_DIR = OUTPUT_DIR / "figures"                  # 图表输出目录

PDF_PATH = DATA_DIR / "招股说明书1.pdf"               # 知识库源文档
QUESTIONS_PATH = DATA_DIR / "questions.json"         # 待评估问题清单

# ---- 模型配置（Ollama 本地模型） ---------------------------------------------
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-r1:1.5b")        # 本地大模型（生成/裁判）
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "bge-m3:567m")  # 本地向量模型

LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "768"))
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "60"))

# ---- 【优化前】基线参数（与 01 工单完全一致，用于 before/after 对比）----------
BASE_CHUNK_SIZE = 500
BASE_CHUNK_OVERLAP = 80
BASE_TOP_K = 6
BASE_BM25_TOP_K = 6
BASE_RRF_K = 60

# ---- 【优化后】分块参数 -------------------------------------------------------
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "700"))       # 结构感知切片大小
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120")) # 切片重叠
MIN_CHUNK_SIZE = int(os.getenv("MIN_CHUNK_SIZE", "60"))# 过短片段合并阈值
USE_PARENT_CHILD = os.getenv("USE_PARENT_CHILD", "1") == "1"  # 父子块（小节级父块）

# ---- 【优化后】检索参数 -------------------------------------------------------
TOP_K = int(os.getenv("TOP_K", "6"))                   # 最终返回片段数
VECTOR_TOP_K = int(os.getenv("VECTOR_TOP_K", "15"))    # 向量召回候选数
BM25_TOP_K = int(os.getenv("BM25_TOP_K", "15"))        # BM25 召回候选数
RANK_FUSION_K = int(os.getenv("RANK_FUSION_K", "60"))  # RRF 窗口
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "6"))     # 重排后保留数
ANSWER_POOL = int(os.getenv("ANSWER_POOL", "25"))      # 答案合成候选池大小（扩大召回）

# 重排权重（加权线性融合，权重之和为 1）
W_VECTOR = float(os.getenv("W_VECTOR", "0.30"))        # 语义相似度
W_BM25 = float(os.getenv("W_BM25", "0.20"))            # 词法命中
W_KEYWORD = float(os.getenv("W_KEYWORD", "0.25"))      # 查询关键词覆盖率
W_NUMERIC = float(os.getenv("W_NUMERIC", "0.15"))      # 数值/年份匹配
W_ENTITY = float(os.getenv("W_ENTITY", "0.10"))        # 实体匹配

# 查询扩展（Query 理解）
USE_QUERY_EXPANSION = os.getenv("USE_QUERY_EXPANSION", "1") == "1"
USE_LLM_QUERY_UNDERSTANDING = os.getenv("USE_LLM_QUERY_UNDERSTANDING", "0") == "1"

# ---- 【优化后】答案生成 -------------------------------------------------------
# extractive：抽取式答案合成（快、稳、可溯源，默认）
# llm：调用本地大模型生成（raw 模式规避思维链）
ANSWER_MODE = os.getenv("ANSWER_MODE", "extractive")
EXTRACTIVE_MAX_SENTENCES = int(os.getenv("EXTRACTIVE_MAX_SENTENCES", "3"))

VECTOR_COLLECTION = os.getenv("VECTOR_COLLECTION", "prospectus_qa_opt")

# ---- 性能目标 -----------------------------------------------------------------
TARGET_RESPONSE_SECONDS = 3.0   # 需求：从提问到生成答案 <= 3s
TARGET_ACCURACY = 0.90          # 需求：问答准确率 >= 90%


def ensure_dirs() -> None:
    """确保关键目录存在（首次运行自动创建）。"""
    for d in (DATA_DIR, DB_DIR, BASE_DB_DIR, OUTPUT_DIR, FIGURE_DIR):
        try:
            d.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass


ensure_dirs()