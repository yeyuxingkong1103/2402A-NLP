# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
模块：全局配置（路径 / 模型 / 参数）

本工单在 01/02 基础上有两个关键变化：
  1. 新增《招股说明书2.pdf》（武汉力源信息技术股份有限公司）→ 多文档问答；
  2. 表格作为一等公民：表格识别 → 结构化 → 入库 → 检索 → 生成全链路专项处理。

硬约束：只使用本机已下载模型，任何路径都不触发下载。
本机环境实测结论见 `docs/01-本机模型扫描结果.md`（继承自工单2）。
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
TABLE_DIR = os.path.join(DATA_DIR, "tables")         # 表格结构化中间产物（工单3新增）
QDRANT_DIR = os.path.join(DATA_DIR, "qdrant")        # Qdrant 本地存储
EVAL_DIR = os.path.join(DATA_DIR, "eval")            # 评估结果
DOCS_DIR = os.path.join(ROOT, "docs")
TESTS_DIR = os.path.join(ROOT, "tests")
LOG_DIR = os.path.join(ROOT, "logs")

# ---------------------------------------------------------------------------
# 多文档（工单3 核心变化之一）
# ---------------------------------------------------------------------------
# 招股说明书1：武汉兴图新科电子股份有限公司（548 页，科创板）
# 招股说明书2：武汉力源信息技术股份有限公司（350 页，创业板）
SOURCE_PDFS = [
    os.path.join(ROOT, "招股说明书1.pdf"),
    os.path.join(ROOT, "招股说明书2.pdf"),
]
SOURCE_NAMES = [os.path.basename(p) for p in SOURCE_PDFS]

# 兼容单文档接口（沿用 01/02 的调用点）
SOURCE_PDF = SOURCE_PDFS[0]
SOURCE_NAME = SOURCE_NAMES[0]

for _d in (DATA_DIR, PARSED_DIR, CHUNK_DIR, TABLE_DIR, QDRANT_DIR, EVAL_DIR,
           DOCS_DIR, TESTS_DIR, LOG_DIR):
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

# [MinerU] 3.4.5 + 独立 venv（2026-10-03 实测打通，工单2 沿用）
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
COLLECTION_OPT = "zhaogu_v3_table"   # 工单3 collection（多文档 + 表格结构化）
VECTOR_DISTANCE = "COSINE"
UPSERT_BATCH_SIZE = 128
UPSERT_PARALLEL = 4

# ---------------------------------------------------------------------------
# 上游系统（对比评估用）—— 只读引用，不修改
# ---------------------------------------------------------------------------
GY1_ROOT = os.path.join(os.path.dirname(ROOT), "工单1")   # 01：单文档基线
GY2_ROOT = os.path.join(os.path.dirname(ROOT), "工单2")   # 02：优化版（无表格结构化）

BASELINE_ROOT = GY1_ROOT                                  # 兼容旧调用点
BASELINE_QDRANT_DIR = os.path.join(GY1_ROOT, "data", "qdrant")
BASELINE_COLLECTION = "zhaogu_shuoming_shu"
BASELINE_CHUNKS = os.path.join(GY1_ROOT, "data", "chunks", "chunks.json")
BASELINE_GT_FILE = os.path.join(GY1_ROOT, "data", "eval", "ground_truth.json")
BASELINE_EVAL_RESULTS = os.path.join(GY1_ROOT, "data", "eval", "eval_results.json")

# 工单2 系统（「无表格优化」对比基线）：仅招股1 有索引
GY2_QDRANT_DIR = os.path.join(GY2_ROOT, "data", "qdrant")
GY2_COLLECTION = "zhaogu_v2_opt"
GY2_PYTHON = r"D:\Python\python.exe"

# ---------------------------------------------------------------------------
# 分块参数（父子块）
# ---------------------------------------------------------------------------
CHILD_MIN_CHARS = 300                # 子块目标下限
CHILD_MAX_CHARS = 450                # 子块目标上限
PARENT_MAX_CHARS = 1800              # 父块（LLM 上下文）上限
TABLE_MAX_CHARS = 3000               # 表格整块保留上限（无结构化时的降级路径）
SECTION_MIN_CHARS = 80               # 过短小节并入相邻

# ---------------------------------------------------------------------------
# 表格参数（工单3 核心）
# ---------------------------------------------------------------------------
TABLE_ENABLED = True                 # 表格结构化总开关（False 时退化为整表块）
TABLE_ROW_MAX_CHARS = 600            # 单行描述上限（超长行截断，避免一条占满向量）
TABLE_CTX_MAX_CHARS = 2600           # 表格上下文（整表）预算
TABLE_MAX_ROWS_PER_TABLE = 400       # 单表最大行数（超长表的保护阈值）
TABLE_MIN_CELLS = 3                  # 少于该单元格数视为低质量表
TABLE_MIN_ALNUM_RATIO = 0.30         # 非空字符占比低于此值视为低质量表
TABLE_OCR_FALLBACK = True            # 低质量表调用 PaddleOCR-VL 兜底
TABLE_QUALITY_PASS = 0.55            # 质量分 ≥ 该值不触发兜底
TABLE_HEADER_MAX_CHARS = 300         # 表头块文本上限

# ---------------------------------------------------------------------------
# 多文档路由（工单3）
# ---------------------------------------------------------------------------
# 问题中出现下列别名时，把检索限定到对应 source
DOC_ALIASES = {
    "招股说明书2.pdf": (
        "武汉力源信息技术股份有限公司", "武汉力源", "力源信息", "力源",
        "赵马克", "Mark Zhao",
    ),
    "招股说明书1.pdf": (
        "武汉兴图新科电子股份有限公司", "武汉兴图新科", "兴图新科", "兴图",
    ),
}
DOC_ROUTER_ENABLED = True
DOC_ROUTER_FALLBACK_OPEN = True      # 过滤后候选不足时放开过滤重检

# ---------------------------------------------------------------------------
# 检索参数
# ---------------------------------------------------------------------------
VECTOR_TOP_K = 30
BM25_TOP_K = 30
RRF_K = 60
HYBRID_ENABLED_DEFAULT = True        # 向量 + BM25 + RRF
RERANK_INPUT_TOP_K = 20              # 融合后进入精排的候选数
# 最终送 LLM 的片段数。
# 工单2 曾用 4（当时是单文档、表格整块，噪声源少）。工单3 实测发现 4 不够：
# 招股书里有大量"**只含问题措辞、不含答案**"的标题/小标题块（如"1、存在控制关系的
# 关联方"），cross-encoder 会给它们 0.95+ 的高分，把真正的答案行（0.94）挤到第 5 位。
# 提到 6 后答案行能进入上下文；CONTEXT_MAX_CHARS 仍会兜住总长度，不会明显拖慢生成。
FINAL_TOP_K = 6
CONTEXT_DEDUP_SAME_PAGE = True       # 同一页的多个片段只保留最佳者（表格块除外）
CONTEXT_MAX_CHARS = 3600             # 送 LLM 上下文总长上限（配合 FINAL_TOP_K=6）

# ---------------------------------------------------------------------------
# 缓存
# ---------------------------------------------------------------------------
ANSWER_CACHE_SIZE = 128
QUERY_ANALYSIS_CACHE_SIZE = 256

# ---------------------------------------------------------------------------
# 评估
# ---------------------------------------------------------------------------
EVAL_QUESTIONS_FILE = os.path.join(DATA_DIR, "test_questions.json")
GT_FILE = os.path.join(EVAL_DIR, "ground_truth.json")
RAGAS_METRICS = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]

WORKORDER_ID = "人工智能NLP-RAG-PDF文档的表格解析及检索优化"
