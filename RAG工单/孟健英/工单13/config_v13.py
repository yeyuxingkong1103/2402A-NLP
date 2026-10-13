# 工单编号：人工智能 NLP-RAG-RAG 性能瓶颈识别与优化
import os

BASE_DIR = r"D:\专高六 工单\工单13"

# 从工单9借用数据
CHROMA_DIR = r"D:\专高六 工单\工单9\rag_project_v9\chroma_db_v9"
CHUNK_FILE = r"D:\专高六 工单\工单9\rag_project_v9\data\chunks_v9.json"
GRAPH_FILE = r"D:\专高六 工单\工单9\rag_project_v9\data\graph_v9.json"

EMBED_MODEL_PATH = r"D:\models（文本嵌入模型）\bge_m3"
RERANK_MODEL_PATH = r"D:\models（文本嵌入模型）\rerank_（重排序 model)\BAAIbge-reranker-base\2cfc18c9415c912f9d8155881c133215df768a70"

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"

QUESTION_FILE = os.path.join(BASE_DIR, "questions_v13.json")
PROFILE_FILE = os.path.join(BASE_DIR, "profile_results_v13.json")
COMPARE_FILE = os.path.join(BASE_DIR, "compare_results_v13.json")