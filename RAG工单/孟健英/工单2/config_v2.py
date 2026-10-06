


# 工单编号：人工智能 NLP-RAG-基于 PDF 文档的问答系统优化
import os

BASE_DIR = r"D:\专高六 工单\工单2"
PDF_PATH = os.path.join(BASE_DIR, "招股说明书1.pdf")

EMBED_MODEL_PATH = r"D:\models（文本嵌入模型）\bge_m3"
RERANK_MODEL_PATH = r"D:\models（文本嵌入模型）\rerank_（重排序 model)\BAAIbge-reranker-base\2cfc18c9415c912f9d8155881c133215df768a70"

CHROMA_DIR = os.path.join(BASE_DIR, "rag_project_v2", "chroma_db_v2")
CHUNK_FILE = os.path.join(BASE_DIR, "rag_project_v2", "data", "chunks_v2.json")

DEEPSEEK_API_KEY = "your-deepseek-api-key"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"

CHUNK_SIZE = 550
CHUNK_OVERLAP = 80
TOP_K = 15
RERANK_TOP = 6