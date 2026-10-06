# 工单编号：人工智能 NLP-RAG-功能测试及评估
import os

BASE_DIR = r"D:\专高六 工单\工单7"
DOCS_DIR = os.path.join(BASE_DIR, "docs")

EMBED_MODEL_PATH = r"D:\models（文本嵌入模型）\bge_m3"
RERANK_MODEL_PATH = r"D:\models（文本嵌入模型）\rerank_（重排序 model)\BAAIbge-reranker-base\2cfc18c9415c912f9d8155881c133215df768a70"

CHROMA_DIR = os.path.join(BASE_DIR, "rag_project_v7", "chroma_db_v7")
CHUNK_FILE = os.path.join(BASE_DIR, "rag_project_v7", "data", "chunks_v7.json")
QUESTION_FILE = os.path.join(BASE_DIR, "questions_v7.json")

DEEPSEEK_API_KEY = "your-deepseek-api-key"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"

CHUNK_SIZE = 550
CHUNK_OVERLAP = 80
TOP_K = 15
RERANK_TOP = 8