# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化
模块：全局配置（路径 / 模型 / 参数）

硬约束：只使用本机已下载模型，任何路径都不触发下载。
选型依据见 `docs/01-本机模型扫描结果.md` 与 `docs/02-优化方案.md`。
"""
from src import bootstrap  # noqa: F401  —— 必须最先导入（离线环境 + 栈修复）

import os

# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------
ROOT = bootstrap.project_root()
DATA_DIR = os.path.join(ROOT, "data")
PARSED_DIR = os.path.join(DATA_DIR, "parsed_mineru")  # MinerU 解析产物（主解析器）
CHUNK_DIR = os.path.join(DATA_DIR, "chunks")         # 分块结果
QDRANT_DIR = os.path.join(DATA_DIR, "qdrant")        # Qdrant 本地存储
EVAL_DIR = os.path.join(DATA_DIR, "eval")            # 评估结果
DOCS_DIR = os.path.join(ROOT, "docs")
TESTS_DIR = os.path.join(ROOT, "tests")

SOURCE_PDF = os.path.join(ROOT, "招股说明书1.pdf")
SOURCE_NAME = os.path.basename(SOURCE_PDF)

for _d in (DATA_DIR, PARSED_DIR, CHUNK_DIR, QDRANT_DIR, EVAL_DIR, DOCS_DIR, TESTS_DIR):
    os.makedirs(_d, exist_ok=True)

# ---------------------------------------------------------------------------
# 模型（全部本机路径，禁止下载）
# ---------------------------------------------------------------------------
# [Embedding] bge-m3：1024 维、中英双语、local_files_only
EMBEDDING_MODEL_PATH = r"D:\model\bge-m3"
EMBEDDING_DIM = 1024
EMBEDDING_DEVICE = "cuda"            # 不可用时自动回退 cpu
EMBEDDING_BATCH_SIZE = 32
EMBEDDING_MAX_LEN = 1024

# [Reranker] bge-reranker-v2-m3（XLMRobertaForSequenceClassification）
RERANKER_MODEL_PATH = r"D:\model\reranker"
RERANKER_ENABLED = True
RERANKER_TOP_K_IN = 20
RERANKER_TOP_K_OUT = 6
RERANKER_MAX_LEN = 512
RERANKER_BATCH_SIZE = 16
RERANKER_TIE_EPSILON = 0.01

# [LLM] Ollama qwen2.5:3b（生成）/ qwen2.5:0.5b（轻量改写、兜底）
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_MODEL = "qwen2.5:3b"
OLLAMA_FALLBACK_MODEL = "qwen2.5:0.5b"
OLLAMA_TEMPERATURE = 0.0            # 事实型问答：贪心解码，最小化随机性/数字幻觉
OLLAMA_NUM_CTX = 8192
OLLAMA_NUM_PREDICT = 512
OLLAMA_KEEP_ALIVE = "30m"            # 常驻显存，避免每题冷启动
LLM_TIMEOUT_S = 120

# [MinerU] 3.4.5 + 独立 venv（2026-10-03 实测打通）
#   版本选择过程（完整记录见 docs/02-优化方案.md 2.1 节）：
#     - 2.7.6 pipeline：需 Layout/YOLO 与 MFD 的 .pt 权重（本机快照缺，禁下载）→ 不可用
#     - 2.7.6 vlm：1.2B VLM 在 8G 显存单页 >1h → 不可用
#     - 4.0.10：本地解析需 GGUF 量化模型（本机无，禁下载）→ 不可用
#     - 3.4.5 pipeline：模型清单与本机 PDF-Extract-Kit 快照**完全匹配**
#       （PP-DocLayoutV2 / unimernet / paddleocr_torch / TabRec / TabCls 全在本地）
#       → ✅ 实测可用：版面分析 ~1.1s/页，全量 548 页约十几分钟
#   依赖隔离：MinerU 3.4.5 需要 transformers 4.x 的旧 API，而主环境为 5.15，
#     故使用独立 venv（--system-site-packages 复用主环境 CUDA torch）。
MINERU_ENABLED = True
MINERU_PYTHON = r"D:\model\mineru-venv\Scripts\python.exe"
MINERU_RUN_SCRIPT = os.path.join(ROOT, "scripts", "mineru_run.py")
MINERU_BACKEND = "pipeline"
MINERU_VLM_MODELS = r"C:\Users\35071\.cache\modelscope\models\OpenDataLab--MinerU2.5-Pro-2605-1.2B\snapshots\master"
MINERU_PIPELINE_MODELS = r"C:\Users\35071\.cache\modelscope\models\OpenDataLab--PDF-Extract-Kit-1.0\snapshots\master"
MINERU_TIMEOUT_S = 7200
MINERU_COMPAT_SCRIPT = os.path.join(ROOT, "scripts", "mineru_compat.py")  # 2.x/vlm 备用

# [OCR 兜底] PaddleOCR-VL 独立 venv，子进程调用（不污染主环境）
PADDLEOCR_PYTHON = r"D:\model\Paddle-OCR\venv\Scripts\python.exe"
PADDLEOCR_VL_PATH = r"D:\model\Paddle-OCR\official_models\PaddleOCR-VL"
OCR_MIN_CHARS_PER_PAGE = 30          # 页面文本低于此值视为需 OCR
OCR_TIMEOUT_S = 600
OCR_MAX_PAGES = 40                   # 单次兜底最多处理页数

# ---------------------------------------------------------------------------
# Qdrant（本机无 Docker：local 模式；起服务后把 QDRANT_MODE 改为 server 即可）
# ---------------------------------------------------------------------------
QDRANT_MODE = "local"                # "local" | "server"
QDRANT_LOCAL_PATH = QDRANT_DIR
QDRANT_URL = "http://127.0.0.1:6333"
COLLECTION_OPT = "zhaogu_v2_opt"     # 优化版 collection
VECTOR_DISTANCE = "COSINE"
UPSERT_BATCH_SIZE = 128
UPSERT_PARALLEL = 4

# ---------------------------------------------------------------------------
# 优化前（工单1）资源 —— 只读引用，不修改
# ---------------------------------------------------------------------------
BASELINE_ROOT = os.path.join(os.path.dirname(ROOT), "工单1")
BASELINE_QDRANT_DIR = os.path.join(BASELINE_ROOT, "data", "qdrant")
BASELINE_COLLECTION = "zhaogu_shuoming_shu"
BASELINE_CHUNKS = os.path.join(BASELINE_ROOT, "data", "chunks", "chunks.json")
BASELINE_GT_FILE = os.path.join(BASELINE_ROOT, "data", "eval", "ground_truth.json")
BASELINE_EVAL_RESULTS = os.path.join(BASELINE_ROOT, "data", "eval", "eval_results.json")

# ---------------------------------------------------------------------------
# 分块参数（父子块）
# ---------------------------------------------------------------------------
CHILD_MIN_CHARS = 300                # 子块目标下限
CHILD_MAX_CHARS = 450                # 子块目标上限
PARENT_MAX_CHARS = 1800              # 父块（LLM 上下文）上限
TABLE_MAX_CHARS = 3000               # 表格整块保留上限
SECTION_MIN_CHARS = 80               # 过短小节并入相邻

# ---------------------------------------------------------------------------
# 检索参数
# ---------------------------------------------------------------------------
VECTOR_TOP_K = 30
BM25_TOP_K = 30
RRF_K = 60
HYBRID_ENABLED_DEFAULT = True        # 向量 + BM25 + RRF
RERANK_INPUT_TOP_K = 20              # 融合后进入精排的候选数
FINAL_TOP_K = 4                      # 最终送 LLM 的子块数（实测 4 在召回与干扰间最平衡；
                                     # 试过 3/6：3 会漏掉"以正文为准"的对照片段，6 噪声过大）
CONTEXT_DEDUP_SAME_PAGE = True       # 同一页的多个片段只保留最佳者，减少重复干扰
CONTEXT_MAX_CHARS = 3200             # 送 LLM 上下文总长上限

# ---------------------------------------------------------------------------
# 缓存
# ---------------------------------------------------------------------------
ANSWER_CACHE_SIZE = 128
QUERY_ANALYSIS_CACHE_SIZE = 256

# ---------------------------------------------------------------------------
# 评估
# ---------------------------------------------------------------------------
EVAL_QUESTIONS_FILE = os.path.join(DATA_DIR, "test_questions.json")
RAGAS_METRICS = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]

WORKORDER_ID = "人工智能NLP-RAG-基于PDF文档的问答系统优化"
