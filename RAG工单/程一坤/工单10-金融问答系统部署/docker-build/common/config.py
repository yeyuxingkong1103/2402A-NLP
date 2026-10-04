# -*- coding: utf-8 -*-
"""
公共配置模块
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
说明：工单01-13 共用的路径、模型、Ollama 服务配置
"""
import os

# ── 目录配置 ────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 程一坤 目录
# 附件目录（工单原始 PDF 位置，按实际机器路径可覆盖）
ATTACH_DIR = os.environ.get(
    "RAG_ATTACH_DIR",
    r"C:\Users\92842\Desktop\RAG 工单\附件",
)
PDF_ZGS1 = os.path.join(ATTACH_DIR, "招股说明书1.pdf")   # 工单01/02/05/06/12 使用
PDF_ZGS2 = os.path.join(ATTACH_DIR, "招股说明书2.pdf")   # 工单03/04/12 使用
CCF_PDF_DIR = os.path.join(ATTACH_DIR, "ccf_competition", "pdf")   # 工单07/08 使用
CCF_TXT_DIR = os.path.join(ATTACH_DIR, "ccf_competition", "txt")   # 工单07 可直接用txt

# 向量索引持久化目录（各工单共用，避免重复向量化）
INDEX_DIR = os.path.join(BASE_DIR, "index_store")

# ── Ollama 服务配置 ─────────────────────────────────────────
OLLAMA_BASE = os.environ.get("OLLAMA_BASE", "http://127.0.0.1:11434")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen2.5:7b-instruct")     # 生成模型
EMBED_MODEL = os.environ.get("EMBED_MODEL", "dengcao/bge-m3:567m") # 向量模型(1024维)
VL_MODEL = os.environ.get("VL_MODEL", "qwen2.5vl:7b")              # 多模态模型(工单04)

# ── 分块 / 检索参数 ─────────────────────────────────────────
CHUNK_SIZE = 500      # 分块目标长度（字符）
CHUNK_OVERLAP = 80    # 相邻分块重叠长度
DEFAULT_TOP_K = 5     # 默认召回条数
EMBED_BATCH = 32      # 向量化批大小
