
# 工单编号：人工智能 NLP-RAG-LightRAG 优化
import os

BASE_DIR = r"D:\专高六 工单\工单12"
DOCS_DIR = os.path.join(BASE_DIR, "docs")
WORKING_DIR = os.path.join(BASE_DIR, "rag_project_v12", "lightrag_storage")

EMBED_MODEL_PATH = r"D:\models（文本嵌入模型）\bge_m3"
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"

QUESTION_FILE = os.path.join(BASE_DIR, "questions_v12.json")
COMPARE_RESULT_FILE = os.path.join(BASE_DIR, "compare_results_v12.json")
EVAL_RESULT_FILE = os.path.join(BASE_DIR, "eval_results_v12.json")
RAG_RESULT_FILE = os.path.join(BASE_DIR, "rag_v12_results.json")

# 只索引前 N 页
MAX_PAGES = 150