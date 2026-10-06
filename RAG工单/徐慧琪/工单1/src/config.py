# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
模块：全局配置

所有路径与模型选型集中在此处。选型依据见下方各常量注释，
原则：**只使用本机已下载的模型，任何路径都不允许触发下载**。
"""

from src import bootstrap  # noqa: F401  —— 必须最先导入（离线环境 + 导入链预热）

import os

# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------
ROOT = bootstrap.project_root()
DATA_DIR = os.path.join(ROOT, "data")
PARSED_DIR = os.path.join(DATA_DIR, "parsed")      # PDF 解析产物（markdown / content_list.json）
CHUNK_DIR = os.path.join(DATA_DIR, "chunks")       # 分块结果
QDRANT_DIR = os.path.join(DATA_DIR, "qdrant")      # Qdrant 本地嵌入式存储
EVAL_DIR = os.path.join(DATA_DIR, "eval")          # 评估结果
DOCS_DIR = os.path.join(ROOT, "docs")

# 待问答的目标文档
SOURCE_PDF = os.path.join(ROOT, "招股说明书1.pdf")

for _d in (DATA_DIR, PARSED_DIR, CHUNK_DIR, QDRANT_DIR, EVAL_DIR, DOCS_DIR):
    os.makedirs(_d, exist_ok=True)

# 文档标识：写入向量 payload 的 source 字段，支持将来多文档知识库
SOURCE_NAME = os.path.basename(SOURCE_PDF)

# ---------------------------------------------------------------------------
# 模型选型（第二步产物）
# ---------------------------------------------------------------------------
# [Embedding] D:/model/bge-m3
#   理由：工单优先级 "BGE 系列（bge-large-zh、bge-m3）" 的首选；
#        1024 维、max_seq 8194、中英双语（工单要求支持中英文问答）；
#        本地为 sentence-transformers 完整目录（含 modules.json / 1_Pooling），可直接加载。
#   加载：local_files_only=True
EMBEDDING_MODEL_PATH = r"D:\model\bge-m3"
EMBEDDING_DIM = 1024
EMBEDDING_DEVICE = "cuda"          # RTX 4060 Laptop 8G；不可用时回退 cpu
EMBEDDING_BATCH_SIZE = 32
EMBEDDING_MAX_LEN = 1024           # 招股书单块较长，1024 足够且显著快于 8192

# [Reranker] D:/model/reranker  —— BAAI/bge-reranker-v2-m3 (XLMRobertaForSequenceClassification)
#   理由：工单列为可选优化项；本地已有，用于对向量召回结果做精排，提升 top-k 精度。
RERANKER_MODEL_PATH = r"D:\model\reranker"
RERANKER_ENABLED = True
RERANKER_TOP_K_IN = 20             # 精排前召回数
RERANKER_TOP_K_OUT = 6             # 精排后送入 LLM 的块数
RERANKER_MAX_LEN = 512             # 精排截断长度；招股书单块 ≤1800 字符，512 token 足够
RERANKER_BATCH_SIZE = 16
RERANKER_TIE_EPSILON = 0.01        # 精排分差值小于此值视为"平局"，回退用融合名次裁决（见 reranker.py）

# [LLM] Ollama qwen2.5:3b
#   理由：工单优先级 "Qwen 系列（qwen2.5-7b/14b）" 的首选家族；
#        本机 Ollama 已 pull 且离线可用（模型清单见 D:\Ollama\OLLAMA_MODELS\manifests），
#        调用已下载模型不触发任何下载，满足硬约束。
#   备选：qwen2.5:0.5b（更快、质量低，仅用于冒烟测试 / 极低配回退）
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_MODEL = "qwen2.5:3b"
OLLAMA_FALLBACK_MODEL = "qwen2.5:0.5b"
OLLAMA_TEMPERATURE = 0.1
OLLAMA_NUM_CTX = 8192              # 需容纳若干检索块 + 回答
OLLAMA_NUM_PREDICT = 512
LLM_TIMEOUT_S = 120

# [OCR] D:/model/Paddle-OCR/official_models/PaddleOCR-VL
#   理由：本机已下载权重目录。招股书全文有文本层（548 页均有可提取文字），
#        故 OCR 仅作为"扫描页兜底"按需启用，不在主链路。
PADDLEOCR_VL_PATH = r"D:\model\Paddle-OCR\official_models\PaddleOCR-VL"
PADDLEOCR_PYTHON = r"D:\model\Paddle-OCR\venv\Scripts\python.exe"
OCR_ENABLED = False                # 主链路关闭；检测到无文本层页面时才启用

# [ASR] 语音输入 —— Windows 系统自带离线识别器（System.Speech / SAPI 5.4）
#   理由：工单"产出物/系统功能 1"要求问答界面支持**语音输入**；而本机
#        无 whisper / faster-whisper / funasr / vosk，缓存里也没有任何 ASR 权重，
#        硬约束又禁止下载模型 —— 因此唯一可行路径是操作系统自带的识别器：
#        零下载、零额外依赖，直接复用 Windows 的 zh-CN 识别引擎。
#   实测（本机 zh-CN 识别器 MS-2052-80-DESK）：日常问句识别准确，
#        如"军用领域的收入是多少""募集资金多少用于补充流动资金"均逐字正确；
#        专有名词会出现同音错字（"兴图"→"信徒"），由 ASR_HOTWORDS 近音纠正兜底。
#   限制：本机只装了 zh-CN 识别器，**英文语音输入不可用**（英文文字问答不受影响）。
ASR_ENABLED = True
ASR_TIMEOUT_S = 30                 # 单次识别墙钟上限
ASR_MAX_RESULTS = 15               # 单次识别最多收集的语音段数
# 领域热词：识别结果里若出现与热词"长度相同、仅个别字不同"的片段，
# 视为同音误识并纠正（见 asr.correct_domain_terms）。
ASR_HOTWORDS = [
    "武汉兴图新科电子股份有限公司",
    "注册资本", "法定代表人", "军用领域", "主营业务收入", "募集资金",
    "补充流动资金", "国家科技进步一等奖", "重要供应商", "技术标准",
    "电子信息行业", "上游", "下游", "招股意向书",
]

# [Vector DB] Qdrant
#   说明：工单原计划 Docker 部署 qdrant/qdrant，但本机无 docker 可执行文件、
#        也没有 qdrant 二进制。改用 qdrant-client 的**本地嵌入式模式**（path=），
#        同一套 API 与数据结构，无需服务端；若日后起了 Qdrant 服务，
#        把 QDRANT_MODE 改成 "server" 并填 URL 即可无缝切换。
QDRANT_MODE = "local"              # "local" | "server"
QDRANT_LOCAL_PATH = QDRANT_DIR
QDRANT_URL = "http://127.0.0.1:6333"
COLLECTION_NAME = "zhaogu_shuoming_shu"
VECTOR_DISTANCE = "COSINE"
UPSERT_BATCH_SIZE = 100
UPSERT_PARALLEL = 4

# ---------------------------------------------------------------------------
# 分块参数
# ---------------------------------------------------------------------------
CHUNK_MAX_CHARS = 900              # 目标块大小（中文字符）
CHUNK_MIN_CHARS = 120              # 低于此长度的碎块并入相邻块
CHUNK_OVERLAP_CHARS = 120          # 相邻块重叠，避免答案被切断
TABLE_MAX_CHARS = 1800             # 表格块上限（表格不切分，超长才截断保留表头）

# ---------------------------------------------------------------------------
# 检索参数
# ---------------------------------------------------------------------------
VECTOR_TOP_K = 20                  # 向量召回数
BM25_TOP_K = 20                    # 稀疏召回数（中文分词）
HYBRID_ENABLED = True              # 向量 + BM25 混合召回（RRF 融合）
FINAL_TOP_K = 6                    # 最终送入 LLM 的块数

# ---------------------------------------------------------------------------
# 评估
# ---------------------------------------------------------------------------
EVAL_QUESTIONS_FILE = os.path.join(DATA_DIR, "test_questions.json")
RAGAS_METRICS = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]

WORKORDER_ID = "人工智能NLP-RAG-基于PDF文档的问答系统"
