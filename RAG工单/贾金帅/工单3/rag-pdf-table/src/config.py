"""
配置中心
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化
工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化

所有可调参数集中在此，读取根目录 .env。其他模块只从这里取常量，不自己读环境变量，
避免「同一参数两处各写一份」导致调参漂移。
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------- 工单标识
# 工单要求：代码注释需包含工单编号。所有模块统一引用这几个常量。
WORK_ORDER_NO = "人工智能NLP-RAG-基于PDF文档的问答系统"                 # 工单 01：功能实现
WORK_ORDER_NO_OPT = "人工智能NLP-RAG-基于PDF文档的问答系统优化"          # 工单 02：检索优化
WORK_ORDER_NO_TABLE = "人工智能NLP-RAG-PDF文档的表格解析及检索优化"      # 工单 03：表格解析
WORK_ORDER_NOS = [WORK_ORDER_NO, WORK_ORDER_NO_OPT, WORK_ORDER_NO_TABLE]

# 交付物命名用的短标签
WORK_ORDER_SHORT = "工单3"

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")


def _get(key: str, default: str = "") -> str:
    return os.getenv(key, default).strip()


def _get_int(key: str, default: int) -> int:
    try:
        return int(_get(key, str(default)))
    except ValueError:
        return default


def _get_float(key: str, default: float) -> float:
    try:
        return float(_get(key, str(default)))
    except ValueError:
        return default


def _get_bool(key: str, default: bool) -> bool:
    return _get(key, str(default)).lower() in ("1", "true", "yes", "on")


# ---------------------------------------------------------------- 应用
APP_NAME = _get("APP_NAME", "PDF-RAG-QA")
APP_HOST = _get("APP_HOST", "0.0.0.0")
APP_PORT = _get_int("APP_PORT", 8010)
LOG_LEVEL = _get("LOG_LEVEL", "INFO")

# ---------------------------------------------------------------- 大模型
LLM_API_KEY = _get("LLM_API_KEY")
LLM_BASE_URL = _get("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
LLM_MODEL = _get("LLM_MODEL", "qwen3.7-flash")
LLM_JUDGE_MODEL = _get("LLM_JUDGE_MODEL", LLM_MODEL)
LLM_TEMPERATURE = _get_float("LLM_TEMPERATURE", 0.1)
LLM_MAX_TOKENS = _get_int("LLM_MAX_TOKENS", 1024)
LLM_TIMEOUT_SECONDS = _get_int("LLM_TIMEOUT_SECONDS", 60)

# ---------------------------------------------------------------- 向量模型
EMBEDDING_MODEL_PATH = _get("EMBEDDING_MODEL_PATH", r"D:\bge-small-zh-v1.5")
EMBEDDING_DIM = _get_int("EMBEDDING_DIM", 512)
EMBEDDING_BATCH_SIZE = _get_int("EMBEDDING_BATCH_SIZE", 32)

# ---------------------------------------------------------------- 路径
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
INDEX_DIR = DATA_DIR / "index"
UPLOAD_DIR = DATA_DIR / "uploads"
STATIC_DIR = ROOT_DIR / "static"
EVAL_DIR = ROOT_DIR / "data" / "eval"
# 验收评测集文件名（工单03 起是 14 题：4 道力源新题 + 10 道兴图旧题）
EVAL_DATASET = _get("EVAL_DATASET", "questions_wot3.json")

PDF_PATH = ROOT_DIR / _get("PDF_PATH", "data/raw/zhaogu_yixiangshu.pdf")

# ---------------------------------------------------------------- 多文档语料（工单03）
# 工单03 往系统里加了第二份招股说明书，语料从「单文档」变成「两文档」。
# 每个文档挂一份「页眉匹配正则」：这两份说明书每页都重复打印
# 「公司名 + 招股(意向)书」，必须在入索引前剔掉，否则会被 BM25 当高频词命中。
# 注意文档名要写全称，因为检索结果与答案引用都要靠它区分「答的是哪家公司」。
DOCS: list[dict[str, str]] = [
    {
        "key": "xingtu",
        "file": "zhaogu_yixiangshu.pdf",
        "name": "武汉兴图新科电子股份有限公司招股意向书",
        "company": "武汉兴图新科电子股份有限公司",
        "header_re": r"^\s*武汉兴图新科电子股份有限公司\s*招股意向书\s*$",
    },
    {
        "key": "liyuan",
        "file": "zhaogu2.pdf",
        "name": "武汉力源信息技术股份有限公司招股说明书",
        "company": "武汉力源信息技术股份有限公司",
        "header_re": r"^\s*武汉力源信息技术股份有限公司\s*招股说明书\s*$",
    },
]

DOCS_BY_KEY: dict[str, dict[str, str]] = {d["key"]: d for d in DOCS}

# 文档消歧用的别名（问句里可能写全称、简称，或只写股票简称）。
# 只在「多文档」语境下使用 —— 单文档时不需要、也不该做这个限定。
DOC_ALIASES: dict[str, list[str]] = {
    "xingtu": ["武汉兴图新科电子股份有限公司", "兴图新科", "兴图"],
    "liyuan": ["武汉力源信息技术股份有限公司", "力源信息", "力源"],
}

# 表格解析（工单03 核心）
# 关闭后 → 表格内容退化成普通文本段落参与分块，即「优化前」的对照实现。
TABLE_PARSE_ENABLED = _get_bool("TABLE_PARSE_ENABLED", True)
# 双线边框缝隙的合并阈值（pt）。实测缝隙 5.4~6.3pt、最小真实列宽 66pt，8pt 留有充分余量。
TABLE_GAP_MERGE_PT = _get_float("TABLE_GAP_MERGE_PT", 8.0)
# 单张表转成一个检索单元时最多输出多少行（防超长表吃爆上下文预算）
TABLE_MAX_ROWS = _get_int("TABLE_MAX_ROWS", 40)

# ---------------------------------------------------------------- 多文档「文档提示」（工单03）
# 工单2 的 DF 过滤会**主动删掉公司全称**（单文档语料里它出现在几乎所有块中，
# 是没有区分度的高频噪音）。语料变成两份文档后，这个删法出了新问题：
#   「武汉兴图新科电子股份有限公司注册资本是多少」
#   → 剔除公司名后只剩「注册资本是多少」
#   → 力源信息那一边的「历次验资情况」段落被召回，兴图的正确答案被挤出候选
# 也就是说，**公司名在两文档语料里恰恰是最强的消歧信号**，不能删。
# 做法：先在**原问题**（未过滤）里识别公司名 → 映射到目标文档 → 把候选池限定在该文档内；
# 检索式本身照旧剔除公司名（保留工单2 的收益）。
DOC_HINT_ENABLED = _get_bool("DOC_HINT_ENABLED", True)

# 索引产物
CHUNKS_JSONL = INDEX_DIR / "chunks.jsonl"      # 分块正文 + 元数据
EMBEDDINGS_NPY = INDEX_DIR / "embeddings.npy"  # 归一化向量矩阵 (N, D)
INDEX_META_JSON = INDEX_DIR / "index_meta.json"  # 索引元信息（模型、维度、构建时间）

# 工单03 对照组索引：**表格不做结构化**的那一份（同一套解析/分块/检索，只差一个开关）
# 存在的理由：工单3 的产出物要求「显示检索到的答案及检索精度」，
# 界面上要能随时切换「表格结构化 / 不结构化」两条链路。
# 若每次现算，光向量化就要 3 分钟，演示时不可接受 → 预先落盘一份。
INDEX_NOTABLE_DIR = DATA_DIR / "index_notable"

# ---------------------------------------------------------------- 分块
CHUNK_SIZE = _get_int("CHUNK_SIZE", 600)
CHUNK_OVERLAP = _get_int("CHUNK_OVERLAP", 80)
CHUNK_MIN_LENGTH = _get_int("CHUNK_MIN_LENGTH", 40)
CHUNK_MAX_LENGTH = _get_int("CHUNK_MAX_LENGTH", 1200)

# ---------------------------------------------------------------- 检索
RETRIEVAL_DENSE_TOP_K = _get_int("RETRIEVAL_DENSE_TOP_K", 20)
RETRIEVAL_SPARSE_TOP_K = _get_int("RETRIEVAL_SPARSE_TOP_K", 20)
RETRIEVAL_FINAL_TOP_K = _get_int("RETRIEVAL_FINAL_TOP_K", 6)
RETRIEVAL_BM25_BETA = _get_float("RETRIEVAL_BM25_BETA", 0.30)
RETRIEVAL_DOC_CONSENSUS_DELTA = _get_float("RETRIEVAL_DOC_CONSENSUS_DELTA", 0.25)
# 文档消歧加成（加法先验）：问题点名某份文档时，该文档的块在排序上获得这个加成。
# 只影响 score（排序），闸门仍只看 evidence —— 「问的是哪一家」不该让离题问题过闸门。
RETRIEVAL_DOC_HINT_DELTA = _get_float("RETRIEVAL_DOC_HINT_DELTA", 0.25)
RETRIEVAL_MIN_EVIDENCE = _get_float("RETRIEVAL_MIN_EVIDENCE", 0.75)

# ---------------------------------------------------------------- Query 理解旁路（工单02 响应时间优化）
# 实测：QU 是一次额外的 LLM 往返，平均 962ms，占总耗时 48%，而检索只要 26ms。
# 对于「主语明确、无指代、无多跳」的直陈式问题，规则归一化后的原问题已经能
# 稳定召回正确答案（见 docs/技术文档.md「优化方案」），此时再跑一次 LLM 改写
# 属于纯浪费。做法：先做一次近乎零成本的规则检索，若首条依据分已过**旁路阈值**
# （远高于闸门阈值，确保有充足余量），直接跳过 LLM QU。
QU_BYPASS_ENABLED = _get_bool("QU_BYPASS_ENABLED", True)
QU_BYPASS_MIN_EVIDENCE = _get_float("QU_BYPASS_MIN_EVIDENCE", 0.88)
# 出现这些特征说明问题存在指代/多跳/比较，必须交给 LLM 理解，不得旁路。
# 列表刻意收窄到「真的会让直通翻车」的词：指代（它/该/上述/前者/后者）、
# 比较（对比/区别）、多跳推理（为什么/如何/怎么）、以及**并列复合**。
# 关于「并列复合」的判据：不能只看「分别」——「收入分别是多少」是单一主题、
# 只是要多个期间的值，直通完全没问题；真正会翻车的是「上游和下游分别是什么」
# 这种**跨主题**问法。所以判据是「连接词 + 分别/各」同时出现。
# 像「哪些」「分别」单独出现不拦 —— 实测它们规则归一化后的首条依据分就在 0.94 以上，
# 拦下来纯粹是白花 900ms。
QU_BYPASS_BLOCK_RE = _get(
    "QU_BYPASS_BLOCK_RE",
    r"它|该[公次项]|上述|前者|后者|对比|区别|为什么|如何|怎么|以及|各是|分别对应|[和与及].{0,12}(分别|各)",
)

# ---------------------------------------------------------------- 功能开关
QUERY_UNDERSTANDING_ENABLED = _get_bool("QUERY_UNDERSTANDING_ENABLED", True)
UPLOAD_MAX_MB = _get_int("UPLOAD_MAX_MB", 200)
# 容错：单次提问的最大字符数（超出直接截断，避免超长输入打爆 LLM 上下文预算）
MAX_QUESTION_CHARS = _get_int("MAX_QUESTION_CHARS", 500)

# ---------------------------------------------------------------- 优化前基线（工单02 对照实现）
# 朴素 RAG 基线用的分块参数，刻意与主线不同：定长滑窗、无标题感知。
NAIVE_CHUNK_SIZE = _get_int("NAIVE_CHUNK_SIZE", 500)
NAIVE_CHUNK_OVERLAP = _get_int("NAIVE_CHUNK_OVERLAP", 50)
NAIVE_CHUNK_MIN_LENGTH = _get_int("NAIVE_CHUNK_MIN_LENGTH", 40)
INDEX_NAIVE_DIR = DATA_DIR / "index_naive"
NAIVE_CHUNKS_JSONL = INDEX_NAIVE_DIR / "chunks.jsonl"
NAIVE_EMBEDDINGS_NPY = INDEX_NAIVE_DIR / "embeddings.npy"
NAIVE_INDEX_META_JSON = INDEX_NAIVE_DIR / "index_meta.json"


def ensure_dirs() -> None:
    """确保运行期需要的目录都存在。"""
    for d in (DATA_DIR, RAW_DIR, PROCESSED_DIR, INDEX_DIR, UPLOAD_DIR, EVAL_DIR,
              INDEX_NAIVE_DIR, INDEX_NOTABLE_DIR):
        d.mkdir(parents=True, exist_ok=True)


def as_public_dict() -> dict:
    """对外暴露的非敏感配置（用于 /config 探针，绝不返回 API Key）。"""
    return {
        "work_order_no": WORK_ORDER_NO,
        "work_order_no_opt": WORK_ORDER_NO_OPT,
        "work_order_no_table": WORK_ORDER_NO_TABLE,
        "app_name": APP_NAME,
        "llm_model": LLM_MODEL,
        "llm_base_url": LLM_BASE_URL,
        "llm_configured": bool(LLM_API_KEY),
        "embedding_model_path": EMBEDDING_MODEL_PATH,
        "embedding_dim": EMBEDDING_DIM,
        "docs": [{"key": d["key"], "name": d["name"]} for d in DOCS],
        "doc_hint": {"enabled": DOC_HINT_ENABLED,
                     "aliases": {k: v for k, v in DOC_ALIASES.items()}},
        "table_parsing": {
            "enabled": TABLE_PARSE_ENABLED,
            "gap_merge_pt": TABLE_GAP_MERGE_PT,
            "max_rows": TABLE_MAX_ROWS,
        },
        "chunk": {
            "size": CHUNK_SIZE,
            "overlap": CHUNK_OVERLAP,
            "min_length": CHUNK_MIN_LENGTH,
            "max_length": CHUNK_MAX_LENGTH,
        },
        "retrieval": {
            "dense_top_k": RETRIEVAL_DENSE_TOP_K,
            "sparse_top_k": RETRIEVAL_SPARSE_TOP_K,
            "final_top_k": RETRIEVAL_FINAL_TOP_K,
            "bm25_beta": RETRIEVAL_BM25_BETA,
            "doc_consensus_delta": RETRIEVAL_DOC_CONSENSUS_DELTA,
            "min_evidence": RETRIEVAL_MIN_EVIDENCE,
        },
        "qu_bypass": {
            "enabled": QU_BYPASS_ENABLED,
            "min_evidence": QU_BYPASS_MIN_EVIDENCE,
        },
        "query_understanding_enabled": QUERY_UNDERSTANDING_ENABLED,
        "naive_baseline": {
            "chunk_size": NAIVE_CHUNK_SIZE,
            "chunk_overlap": NAIVE_CHUNK_OVERLAP,
        },
    }
