# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-混合检索任务
（工单 PDF 原文：人工智能NLP-RAG-混合检索任务）
模块：全局配置（路径 / 模型 / 检索策略 / 参数）

本工单（06 混合检索）在工单 01~05 基础上新增**三种可配置的检索策略**：
  · 向量检索（召回 + Reranker 精排）        —— src/strategies.py
  · 全文检索（自研倒排索引：布尔/短语/模糊） —— src/fulltext.py
  · 混合检索（两路融合 + Reranker 精排）     —— src/strategies.py + src/fusion.py
  · Reranker 精排模型注册表（≥2 种可切换）   —— src/rerankers.py

上游能力复用（工单 01~05，均已完成为「优化前」版本，本项目只读引用）：
  01 单文档问答基线 / 02 系统优化 / 03 表格解析与检索优化 /
  04 图像内容解析（VLM 描述缓存随工程落盘，服务 id5/id6 图像题）/
  05 Query 理解（多轮对话、指代消解）

硬约束：只使用本机已下载模型，任何路径都不触发下载（见 src/bootstrap.py）。
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
# [Embedding] 多嵌入模型注册表（工单要求：支持多种嵌入模型并可切换）。
# 说明：不同模型的向量不同，各自对应**独立 collection**（见 collection_name()）；
#      键名同时作为界面与配置里的取值。dim 用于建库校验。
EMBEDDING_MODELS: dict[str, dict] = {
    "bge-m3": {
        "path": r"D:\model\bge-m3",
        "dim": 1024,
        "desc": "BGE-M3（默认）：中英双语、1024 维、8k 上下文，本机主索引",
    },
    "bge-m3-ft-v1": {
        "path": r"D:\model\bge-m3-ft-v1",
        "dim": 1024,
        "desc": "BGE-M3 领域微调版 v1（对比用，需单独建库）",
    },
    "miniLM-L6": {
        "path": os.path.join(
            os.environ.get("USERPROFILE", os.path.expanduser("~")),
            ".cache", "huggingface", "hub",
            "models--sentence-transformers--all-MiniLM-L6-v2", "snapshots",
            "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"),
        "dim": 384,
        "desc": "all-MiniLM-L6-v2（英文小模型，384 维，多模型支持演示用）",
    },
}
EMBEDDING_MODEL_NAME = "bge-m3"      # 默认嵌入模型（可切换）
EMBEDDING_MODEL_PATH = EMBEDDING_MODELS[EMBEDDING_MODEL_NAME]["path"]  # 兼容旧调用
EMBEDDING_DIM = EMBEDDING_MODELS[EMBEDDING_MODEL_NAME]["dim"]
EMBEDDING_DEVICE = "cuda"            # 不可用时自动回退 cpu
EMBEDDING_BATCH_SIZE = 32
EMBEDDING_MAX_LEN = 1024

# [Reranker] 精排模型注册表（工单要求：≥2 种 Reranker 模型，支持切换）
#   · bge-reranker-v2-m3  —— XLMRobertaForSequenceClassification，1024 隐层，本机两大路径
#   · bge-reranker-base   —— 上游工单04 随工程落盘（本工单已复制到 data/models/）
RERANKER_MODELS: dict[str, dict] = {
    "bge-reranker-v2-m3": {
        "path": r"D:\model\reranker",
        "desc": "BGE-Reranker-v2-M3（默认，Large 级）：精排主力，中英双语",
    },
    "bge-reranker-v2-m3-ms": {
        "path": r"D:\model\reranker_ms",
        "desc": "BGE-Reranker-v2-M3（ModelScope 副本）：与上者同构，用于多路径容灾",
    },
    "bge-reranker-base": {
        "path": os.path.join(ROOT, "data", "models", "bge-reranker-base"),
        "desc": "BGE-Reranker-Base：768 隐层轻量精排，CPU 也可跑",
    },
}
RERANKER_MODEL_NAME = "bge-reranker-v2-m3"     # 默认精排模型（可切换）
RERANKER_MODEL_PATH = RERANKER_MODELS[RERANKER_MODEL_NAME]["path"]  # 兼容旧调用
RERANKER_ENABLED = True
RERANKER_TOP_K_IN = 20
RERANKER_TOP_K_OUT = 6
RERANKER_MAX_LEN = 512
RERANKER_BATCH_SIZE = 16
RERANKER_TIE_EPSILON = 0.01
# 精排算法（工单要求：生产默认走 Reranker 模型；TF-IDF / 自适应为用户可选的对比基线）
#   model    —— Reranker 模型精排（生产默认，真实加载推理）
#   llm      —— 基于 LLM 的重排器（Ollama qwen2.5:3b 逐条打分）
#   tfidf    —— 基于 TF-IDF 的重排器（对比基线，非生产）
#   adaptive —— 基于用户反馈的自适应重排器（对比基线，吸收点赞/点踩）
RERANK_ALGORITHM_DEFAULT = "model"
RERANK_ALGORITHMS = ("model", "llm", "tfidf", "adaptive")
RERANK_TIMEOUT_S = 8.0               # 单次重排超时（含 LLM 重排）；超时降级
LLM_RERANK_POOL = 20                 # LLM 重排最多逐条打分的候选数（控制耗时）
FEEDBACK_STORE = os.path.join(DATA_DIR, "feedback", "feedback.jsonl")  # 用户反馈留痕

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
COLLECTION_BASE = "zhaogu_v6_hybrid"  # 工单6 collection 前缀（按嵌入模型派生）
COLLECTION_OPT = COLLECTION_BASE      # 默认 collection（= bge-m3 对应库）
VECTOR_DISTANCE = "COSINE"
UPSERT_BATCH_SIZE = 128
UPSERT_PARALLEL = 4


def collection_name(embedding_model: str | None = None) -> str:
    """嵌入模型 → collection 名。

    为什么一个模型一个库：不同嵌入模型的向量空间不同（维度可能也不同），
    混在一个集合里检索会得到无意义的相似度。默认模型沿用基名，
    其它模型加后缀区分（如 zhaogu_v6_hybrid__bge-m3-ft-v1）。
    """
    name = embedding_model or EMBEDDING_MODEL_NAME
    return COLLECTION_BASE if name == EMBEDDING_MODEL_NAME else f"{COLLECTION_BASE}__{name}"

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

# 工单3 系统（本工单的「优化前」基线）：只读引用，不修改
GY3_ROOT = os.path.join(os.path.dirname(ROOT), "工单3")
GY3_COLLECTION = "zhaogu_v3_table"
GY3_QDRANT_DIR = os.path.join(GY3_ROOT, "data", "qdrant")

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
# 工单6 核心：三种检索策略配置（可在界面/代码中随时切换）
# ---------------------------------------------------------------------------
# strategy：vector | fulltext | hybrid
RETRIEVAL_STRATEGY_DEFAULT = "hybrid"
RETRIEVAL_STRATEGIES = {
    "vector": "向量检索（召回 + Reranker 精排）",
    "fulltext": "全文检索（自研倒排索引：布尔/短语/模糊 + 可选精排）",
    "hybrid": "混合检索（向量 + 全文 融合 + Reranker 精排，推荐）",
}
# 召回候选数（工单配置项：如 top_k_recall=20）
TOP_K_RECALL = 20
# 精排后返回数（工单配置项：如 top_k_rerank=5）
TOP_K_RERANK = 5
# 精排分数阈值：低于该值的候选不返回（0 表示不启用）
SCORE_THRESHOLD = 0.0
# 混合检索两路权重（0~1，界面可调；归一化口径见 RetrievalConfig.normalized_weights）
# 默认等权：RRF 下即上游验证过的"无权重 RRF(k=60)"，保证与 01~05 基线可比。
VECTOR_WEIGHT = 0.5
FULLTEXT_WEIGHT = 0.5
# 融合算法：rrf（倒数排名融合）| weighted（加权平均）| vote（投票/Borda）
FUSION_METHOD_DEFAULT = "rrf"
# 融合截断时**每路保底进入精排**的条数（0=关闭）。
# 实测：id260 正确块只被全文臂命中（RRF≈0.014），被大量"两路都命中"的块挤出
# Top-20，精排看不到 → 答案漏数字。首版保底 5 名仍不够（该块在全文臂排第 12），
# 提到 10 名后覆盖。代价仅是精排候选变多（见 RERANK_MAX_IN）。
ARM_QUOTA = 20
# 精排输入上限：融合 Top-K ∪ 每路保底 去重后的条数上限（0=不限）。
# 实测（id260）：quota=10/cap=40 时候选仅 20 条、正确块仍被去重挤掉；
# quota=20/cap=60 时候选 39 条、正确块升到精排第 1，整链 299ms（精排 249ms）。
RERANK_MAX_IN = 60
FUSION_METHODS = ("rrf", "weighted", "vote")
# 各通路独立召回上限（宽召回，再截断到 TOP_K_RECALL 送精排）
VECTOR_TOP_K = 30
BM25_TOP_K = 30
FULLTEXT_TOP_K = 30
RRF_K = 60
HYBRID_ENABLED_DEFAULT = True        # 兼容旧调用点（= strategy 取 hybrid）
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
# 多轮对话（工单5 核心）
# ---------------------------------------------------------------------------
DIALOGUE_ENABLED = True
DIALOGUE_HISTORY_WINDOW = 3          # 送 LLM 的最近轮数（滑动窗口）
DIALOGUE_MAX_TURNS = 50              # 单会话硬上限
DIALOGUE_SUMMARY_MAX_CHARS = 400     # 更早轮次的摘要上限
DIALOGUE_MAX_SESSIONS = 64           # 会话表 LRU 上限

REWRITE_ENABLED = True
REWRITE_MODEL_JUDGE = "qwen2.5:0.5b"   # L2 实体判定（实测：受约束子任务可靠，0.05-0.25s）
REWRITE_MODEL_GEN = "qwen2.5:3b"       # L3 生成（实测：省略补全只有 3b 能做，~0.49s）
REWRITE_MAX_CHARS = 120                # 改写结果长度上限
REWRITE_TIMEOUT_S = 8                  # 单次改写超时；超时走 L4 规则兜底
REWRITE_NUM_PREDICT = 64               # 改写只输出一句话；默认 512 会白白拖长 L3
REWRITE_CACHE_SIZE = 256
REWRITE_NGRAM_REPEAT_THRESHOLD = 0.5   # 与指令模板的 4-gram 重合率超此值判为「复述指令」

# ---------------------------------------------------------------------------
# 组织结构图（工单5 Q5）
# ---------------------------------------------------------------------------
ORGCHART_ENABLED = True
ORGCHART_SOURCE = "招股说明书2.pdf"
ORGCHART_PAGE_IDX = 38               # 0 基索引；展示为第 39 页（PDF 阅读器页码）
ORGCHART_ANCHOR = "发行人内部组织结构图"   # 用于定位的锚点文本（在 page_idx=37）
ORGCHART_RENDER_DPI = 210
ORGCHART_MERGE_GAP_PT = 5.0          # 文字合并的间距阈值
ORGCHART_LINE_TOL_PT = 3.0           # 连线端点吸附到框边的容差
ORGCHART_POINT_TOL_PT = 2.0          # 端点重合的容差

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

WORKORDER_ID = "人工智能NLP-RAG-混合检索任务"
WORKORDER_ID_PDF = "人工智能NLP-RAG-混合检索任务"
