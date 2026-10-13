# 工单编号：人工智能 NLP-RAG-Embedding 模型微调任务
import os

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

BASE_DIR = r"D:\专高六 工单\工单11"
CHUNK_FILE = os.path.join(BASE_DIR, "rag_project_v9", "data", "chunks_v9.json")
DATASET_FILE = os.path.join(BASE_DIR, "dataset_v11.json")
MODEL_OUTPUT_DIR = os.path.join(BASE_DIR, "models", "bge_base_finetuned")
EVAL_RESULT_FILE = os.path.join(BASE_DIR, "eval_results_v11.json")

BASE_MODEL = r"D:\models（文本嵌入模型）\bge_m3"

MAX_QUERIES = 800
NEG_PER_QUERY = 1

DEEPSEEK_MODEL = "deepseek-chat"

# GPU 微调参数
EPOCHS = 1
BATCH_SIZE = 8
LEARNING_RATE = 5e-6
DEVICE = "cpu"
MAX_SEQ_LENGTH = 256    # 限制序列长度，省显存