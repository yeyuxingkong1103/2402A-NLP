# -*- coding: utf-8 -*-
"""
法律RAG问答系统 - 统一配置文件
所有可调参数集中于此，build_kb.py / rag_chain.py / web_app.py 共享
"""
from pathlib import Path

# ======================== 路径配置 ========================
PROJECT_DIR = Path(__file__).parent
PDF_PATH = PROJECT_DIR / "中国裁判文书网法律数据集_500条.pdf"
# Milvus本地数据库（使用TEMP目录的ASCII路径，避免中文路径和权限问题）
import tempfile
MILVUS_DB = str(Path(tempfile.gettempdir()) / "milvus_legal_rag.db")
BM25_INDEX_PATH = PROJECT_DIR / "bm25_index.pkl"

# ======================== Milvus配置 ========================
COLLECTION_NAME = "legal_cases_rag"        # 法律文书向量集合
HISTORY_COLLECTION = "legal_chat_history"   # 多轮对话历史集合
EMBED_DIM = 1024                            # bge-m3输出维度
IVF_NLIST = 16                            # IVF聚类中心数
IVF_NPROBE = 16                            # 查询时探测聚类数
METRIC_TYPE = "COSINE"                     # 相似度度量

# ======================== 模型配置（Ollama） ========================
OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "bge-m3:567m"    # 嵌入模型，1024维
LLM_MODEL = "qwen2.5:1.5b"    # 对话模型
LLM_TEMPERATURE = 0.3          # 生成温度

# ======================== 文本分块配置 ========================
CHUNK_SIZE = 500               # 每块最大字符数
CHUNK_OVERLAP = 80             # 块间重叠字符数
PARSER_METHOD = "auto"         # PDF解析方法: auto/pymupdf/paddleocr/mineru

# ======================== 检索配置 ========================
TOP_K_DENSE = 20               # 稠密检索返回数
TOP_K_SPARSE = 20              # 稀疏检索返回数
TOP_K_FINAL = 5                # 重排序后最终返回数
RRF_K = 60                     # RRF融合常数
RERANK_CANDIDATES = 30         # 重排序候选上限
RERANKER_MODEL = "BAAI/bge-reranker-base"  # CrossEncoder模型

# ======================== 多轮对话配置 ========================
REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 0
HISTORY_TURNS = 5              # Redis保存最近对话轮数
HISTORY_RECALL_K = 3          # Milvus语义召回Top-K

# ======================== Web服务配置 ========================
WEB_HOST = "localhost"
WEB_PORT = 8000
WEB_SESSION_ID = "web_session" # 固定会话ID

# ======================== 日志配置 ========================
LOG_DIR = PROJECT_DIR / "logs"             # 日志目录（自动创建）
LOG_FILE = LOG_DIR / "app.log"             # 主日志文件
LOG_LEVEL = "INFO"                         # 全局日志级别 DEBUG/INFO/WARNING/ERROR
LOG_FILE_MAX_BYTES = 5 * 1024 * 1024      # 单个日志文件最大 5MB
LOG_FILE_BACKUP_COUNT = 5                 # 保留 5 个历史日志文件
LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
LOG_DATE_FMT = "%Y-%m-%d %H:%M:%S"
LOG_TAIL_DEFAULT = 200                     # /logs 接口默认返回的尾部行数
