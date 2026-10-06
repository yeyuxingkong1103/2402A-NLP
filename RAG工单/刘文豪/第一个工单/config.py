# -*- coding: utf-8 -*-
"""
工单编号：人工智能 NLP-RAG-基于 PDF 文档的问答系统
全局配置：路径、模型、检索参数
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------- 数据 ----------
PDF_PATH = r"D:\桌面\RAG 工单\RAG 工单\附件\招股说明书1-无水印.pdf"  # 同《招股说明书1.pdf》，去水印版文本更干净
DATA_DIR = os.path.normpath(os.path.join(BASE_DIR, "data"))          # 本工单数据目录（固定，不接收外部输入）
PAGES_JSON = os.path.join(DATA_DIR, "pages.json")

# ---------- 向量库 ----------
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")
COLLECTION = "zhaogu1_v1"

# ---------- Ollama 本地模型（127.0.0.1:11434，已在本机运行） ----------
OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "bge-m3"        # 向量化模型（本地 1.2GB）
GEN_MODEL = "qwen2.5:7b"      # 生成模型（本地 4.7GB）

# ---------- 分块 ----------
CHUNK_SIZE = 600              # 每块字符数
CHUNK_OVERLAP = 120           # 相邻块重叠

# ---------- 检索 ----------
TOP_K = 5                     # 召回条数
