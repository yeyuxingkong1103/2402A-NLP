# -*- coding: utf-8 -*-
"""
工单十四：配置模块
"""
import os
from dotenv import load_dotenv

load_dotenv()

# 工单信息
WORK_ORDER = "人工智能NLP-RAG-修复低质量工业PDF的解析与信息丢失"
WORK_ORDER_NO = "工单十四"

# Milvus
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = int(os.getenv("MILVUS_PORT", "19530"))
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "rag_v14_chunks")
MILVUS_LITE_PATH = "./data/milvus_v14.db"

# Embedding
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))
EMBED_BATCH_SIZE = int(os.getenv("RAG_EMBED_BATCH_SIZE", "8"))
EMBED_MAX_SEQ = int(os.getenv("RAG_EMBED_MAX_SEQ", "512"))
EMBED_DEVICE = os.getenv("RAG_EMBED_DEVICE", "cpu")

# LLM
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")

# Reranker
RERANKER_PATH = os.getenv("RERANKER_PATH", "/home/dabaie/models/bge-reranker-v2-m3")

# VLM
VLM_MODEL_DIR = os.getenv("VLM_MODEL_DIR", "/home/dabaie/models/Qwen2-VL-2B-Instruct")
VLM_DEVICE = os.getenv("VLM_DEVICE", "cuda")

# 应用
APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT = int(os.getenv("APP_PORT", "8014"))
STREAMLIT_PORT = int(os.getenv("STREAMLIT_PORT", "8514"))

# 数据路径
PDF_DIR = os.getenv("PDF_DIR", "./附件")
PARSED_DIR = os.getenv("PARSED_DIR", "./data/parsed")
CHUNKS_DIR = os.getenv("CHUNKS_DIR", "./data/chunks")
LOG_DIR = os.getenv("LOG_DIR", "./data/logs")
