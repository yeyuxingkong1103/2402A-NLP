# -*- coding: utf-8 -*-
# 工单编号：人工智能 NLP-RAG-基于 PDF 文档的问答系统优化
# 模块：config —— 全局配置
# 说明：路径、模型、检索与生成参数集中管理，便于换机型/换模型不改业务代码。

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------- 路径 ----------
DATA_DIR = os.path.join(BASE_DIR, "data")
PDF_DIR = os.path.join(DATA_DIR, "pdfs")
MINERU_DIR = os.path.join(DATA_DIR, "mineru")          # MinerU 原始输出根目录
PARSED_DIR = os.path.join(DATA_DIR, "parsed")          # 解析后的统一 blocks
INDEX_DIR = os.path.join(DATA_DIR, "index")            # 向量 + BM25 索引
STATIC_DIR = os.path.join(BASE_DIR, "static")

DEFAULT_PDF = os.path.join(PDF_DIR, "招股说明书1.pdf")

BLOCKS_PATH = os.path.join(PARSED_DIR, "blocks.jsonl")
CHUNKS_PATH = os.path.join(INDEX_DIR, "chunks.jsonl")
EMB_PATH = os.path.join(INDEX_DIR, "embeddings.npy")
BM25_PATH = os.path.join(INDEX_DIR, "bm25.json")
META_PATH = os.path.join(INDEX_DIR, "meta.json")

# ---------- 模型（本机 Ollama） ----------
OLLAMA_BASE = os.environ.get("OLLAMA_BASE", "http://localhost:11434")
EMBED_MODEL = os.environ.get("RAG_EMBED_MODEL", "bge-m3")   # 1024 维，多语
GEN_MODEL = os.environ.get("RAG_GEN_MODEL", "qwen2:7b")     # 生成模型（RAG 与纯 LLM 用同一个，保证对比公平）

# ---------- 分块 ----------
CHUNK_SIZE = 600
CHUNK_OVERLAP = 120
MIN_CHUNK = 40

# ---------- 检索 ----------
RECALL_K = 20          # 每路召回条数
RRF_K = 60             # RRF 平滑常数
RRF_LAMBDA = 2.5       # 向量路权重（BM25 路权重 = 1）
FUSION_MODE = "rrf"    # 融合方式: "rrf" 或 "sum"（归一化加权求和）；实测 rrf/λ=2.5 最优
DENSE_W = 0.6          # sum 模式下向量路权重
SPARSE_W = 0.4         # sum 模式下 BM25 路权重
TOP_K = 5              # 送进生成的片段数
REFUSE_SCORE = 0.58    # 置信度（最高余弦）阈值：低于则拒答。实测标定：库内最低 0.644 / 库外最高 0.509
MAX_CONTEXT_CHARS = 4000

# ---------- 服务 ----------
HOST = "0.0.0.0"
PORT = 8100
