# 工单编号：人工智能 NLP-RAG-混合检索任务
import os

BASE_DIR = r"D:\专高六 工单\工单6"
PDF_LIST = {
    "xingtu": os.path.join(BASE_DIR, "招股说明书1.pdf"),
    "liyuan": os.path.join(BASE_DIR, "招股说明书2.pdf"),
}

EMBED_MODEL_PATH = r"D:\models（文本嵌入模型）\bge_m3"
RERANK_MODEL_PATH = r"D:\models（文本嵌入模型）\rerank_（重排序 model)\BAAIbge-reranker-base\2cfc18c9415c912f9d8155881c133215df768a70"

CHROMA_DIR = os.path.join(BASE_DIR, "rag_project_v6", "chroma_db_v6")
CHUNK_FILE = os.path.join(BASE_DIR, "rag_project_v6", "data", "chunks_v6.json")
IMAGE_DIR = os.path.join(BASE_DIR, "rag_project_v6", "images")
FEEDBACK_FILE = os.path.join(BASE_DIR, "rag_project_v6", "data", "feedback_v6.json")

ZHIPU_API_KEY = "b28f43c088274a838d0cc5d67bcb61ca.xsZy74d0jx9pCN8B"
ZHIPU_MODEL = "glm-4v-flash"

DEEPSEEK_API_KEY = "your-deepseek-api-key"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"

CHUNK_SIZE = 550
CHUNK_OVERLAP = 80
TOP_K = 15
RERANK_TOP = 8
MIN_IMG_W = 200
MIN_IMG_H = 200

# 混合检索默认权重
DEFAULT_VECTOR_WEIGHT = 0.6
DEFAULT_FULLTEXT_WEIGHT = 0.4