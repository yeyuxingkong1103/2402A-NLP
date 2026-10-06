# 工单编号：人工智能 NLP-RAG-金融问答系统部署
import os

BASE_DIR = os.environ.get("RAG_BASE", "/app")
DOCS_DIR = os.path.join(BASE_DIR, "docs")

EMBED_MODEL_PATH = os.environ.get("EMBED_MODEL_PATH", "/models/bge_m3")
RERANK_MODEL_PATH = os.environ.get("RERANK_MODEL_PATH", "/models/bge-reranker-base")

CHROMA_DIR = os.path.join(BASE_DIR, "rag_project_v9", "chroma_db_v9")
CHUNK_FILE = os.path.join(BASE_DIR, "rag_project_v9", "data", "chunks_v9.json")
GRAPH_FILE = os.path.join(BASE_DIR, "rag_project_v9", "data", "graph_v9.json")
QUESTION_FILE = os.path.join(BASE_DIR, "questions_v9.json")

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

CHUNK_SIZE = 700
CHUNK_OVERLAP = 60
TOP_K = 10
RERANK_TOP = 5
GRAPH_HOPS = 1