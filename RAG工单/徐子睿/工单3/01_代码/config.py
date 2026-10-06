# -*- coding: utf-8 -*-
# 工单编号：人工智能 NLP-RAG-基于 PDF 文档的问答系统优化
# 关联工单：人工智能NLP-RAG-PDF文档的表格解析及检索优化 | 人工智能NLP-RAG-图像内容解析及检索优化 | 人工智能NLP-RAG-Query理解优化任务 | 人工智能NLP-RAG-混合检索任务
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

# ---------- 多文档（03/04 工单：在《招股说明书1.pdf》基础上新增《招股说明书2.pdf》） ----------
def list_pdfs():
    """data/pdfs 下的全部 PDF（按文件名排序）。"""
    import glob
    return sorted(glob.glob(os.path.join(PDF_DIR, "*.pdf")))


def doc_of(pdf_path):
    """文档 id = 文件名去扩展名（如 招股说明书1 / 招股说明书2）。"""
    return os.path.splitext(os.path.basename(pdf_path))[0]


def blocks_path(doc):
    """每本 PDF 各自的解析块文件（避免多文档相互覆盖）。"""
    return os.path.join(PARSED_DIR, "blocks_%s.jsonl" % doc)


# 文档别名：问题里出现的公司名/简称/关键词 → 文档 id，用于把检索限定到正确文档（多文档互不串味）
DOC_ALIASES = {
    "招股说明书1": ["兴图新科", "武汉兴图新科", "招股说明书1", "兴图"],
    "招股说明书2": ["力源信息", "武汉力源", "招股说明书2", "力源", "LiYuan"],
}

# ---------- 混合检索（工单 06：人工智能NLP-RAG-混合检索任务） ----------
# 可配置的嵌入模型（键=显示名，值=Ollama 模型名或本地模型路径）
EMBED_MODELS = {
    "bge-m3": "bge-m3",          # 多语稠密，当前默认
    "m3e-base": "m3e-base",      # 中文 m3e（如本地已下拉/加载）
}
DEFAULT_EMBED = os.environ.get("RAG_EMBED", "bge-m3")

# 检索策略：vector（纯向量）/ fulltext（纯全文）/ hybrid（混合）/ rrf（向量+BM25 加权 RRF）
RETRIEVAL_STRATEGY = os.environ.get("RAG_STRATEGY", "hybrid")
HYBRID_WEIGHT_VEC = 0.6       # 混合检索里向量路权重（可调）
HYBRID_WEIGHT_FT = 0.4        # 混合检索里全文路权重（可调）
FUSION_ALGO = "weighted"      # 融合算法：weighted（加权平均）/ voting（投票）/ rrf
RERANKER = os.environ.get("RAG_RERANKER", "none")   # none / llm / tfidf / feedback


# ---------- 服务 ----------
HOST = "0.0.0.0"
PORT = 8100
