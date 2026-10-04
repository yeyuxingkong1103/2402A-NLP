# 工单编号：人工智能 NLP-RAG-图像内容解析及检索优化
import os

BASE_DIR = r"D:\专高六 工单\工单4"
PDF_LIST = {
    "xingtu": os.path.join(BASE_DIR, "招股说明书1.pdf"),
    "liyuan": os.path.join(BASE_DIR, "招股说明书2.pdf"),
}

EMBED_MODEL_PATH = r"D:\models（文本嵌入模型）\bge_m3"
RERANK_MODEL_PATH = r"D:\models（文本嵌入模型）\rerank_（重排序 model)\BAAIbge-reranker-base\2cfc18c9415c912f9d8155881c133215df768a70"

CHROMA_DIR = os.path.join(BASE_DIR, "rag_project_v4", "chroma_db_v4")
CHUNK_FILE = os.path.join(BASE_DIR, "rag_project_v4", "data", "chunks_v4.json")
IMAGE_DIR = os.path.join(BASE_DIR, "rag_project_v4", "images")

# 图像解析用智谱 GLM-4V
ZHIPU_API_KEY = "b28f43c088274a838d0cc5d67bcb61ca.xsZy74d0jx9pCN8B"
ZHIPU_MODEL = "glm-4v-flash"

# 文本生成用 DeepSeek
DEEPSEEK_API_KEY = "your-deepseek-api-key"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"

CHUNK_SIZE = 550
CHUNK_OVERLAP = 80
TOP_K = 15
RERANK_TOP = 8

# 图像过滤：小于该尺寸的图跳过（去 logo、装饰图）
MIN_IMG_W = 200
MIN_IMG_H = 200