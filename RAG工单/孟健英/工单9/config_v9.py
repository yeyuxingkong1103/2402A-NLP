# 工单编号：人工智能 NLP-RAG-Graph RAG 优化任务
import os

BASE_DIR = r"D:\专高六 工单\工单9"
DOCS_DIR = os.path.join(BASE_DIR, "docs")

EMBED_MODEL_PATH = r"D:\models（文本嵌入模型）\bge_m3"
RERANK_MODEL_PATH = r"D:\models（文本嵌入模型）\rerank_（重排序 model)\BAAIbge-reranker-base\2cfc18c9415c912f9d8155881c133215df768a70"

CHROMA_DIR = os.path.join(BASE_DIR, "rag_project_v9", "chroma_db_v9")
CHUNK_FILE = os.path.join(BASE_DIR, "rag_project_v9", "data", "chunks_v9.json")
GRAPH_FILE = os.path.join(BASE_DIR, "rag_project_v9", "data", "graph_v9.json")
QUESTION_FILE = os.path.join(BASE_DIR, "questions_v9.json")

DEEPSEEK_API_KEY = "your-deepseek-api-key"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"

CHUNK_SIZE = 700
CHUNK_OVERLAP = 60
TOP_K = 10
RERANK_TOP = 8
GRAPH_HOPS = 1