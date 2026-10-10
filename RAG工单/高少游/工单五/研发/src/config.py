# -*- coding: utf-8 -*-
"""项目全局配置。

工单编号: 人工智能 NLP-RAG-Query 理解优化任务

本工单在《01 基于 PDF 文档的问答系统》《02 基于 PDF 文档的问答系统的优化》
《03 PDF 文档的表格解析及检索优化》《04 图像内容解析及检索优化》的基础上，
新增 **多轮对话（Multi-turn Dialogue）** 与 **Query 理解优化** 能力：

    - 会话状态管理：记录话题、实体、上一轮意图（conversation.py）
    - 多轮 Query 改写：指代消解（他 / 这个公司）+ 省略补全（那 X 呢？）（query_rewriting.py）
    - 上下文感知检索：改写后的独立问句 + 话题实体加权（qa_engine.py）
    - 同义词扩展：跨表述召回（军用领域 = 国防客户 / 军方市场）（query_understanding.py）
    - 图形语义：组织结构图 → 可检索结构化文本（figure_semantics.py）

所有可调参数优先读取环境变量 / .env，其次使用默认值。
"""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ---- 目录与路径 -------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent      # 项目根目录（研发/）
DATA_DIR = BASE_DIR / "data"                           # 数据目录（PDF、题目）
DB_DIR = BASE_DIR / "vector_db"                        # 轻量向量库目录
OUTPUT_DIR = BASE_DIR / "output"                       # 评估结果 / 日志输出目录
FIGURE_OUT_DIR = OUTPUT_DIR / "figures"                # 评估图表输出目录
SHOT_DIR = OUTPUT_DIR / "shots"                        # 界面截图输出目录

CHUNKS_PATH = DB_DIR / "chunks.jsonl"                  # 切片文本（含元数据）
EMB_PATH = DB_DIR / "embeddings.npy"                   # 切片向量（float32）
META_PATH = DB_DIR / "meta.json"                       # 向量库元信息
QUESTIONS_PATH = DATA_DIR / "questions.json"           # 单轮验收问题清单
MULTITURN_PATH = DATA_DIR / "multiturn_dialogue.json"  # 多轮对话验收脚本

# 知识库源文档：兴图新科（招股说明书1） + 力源信息（招股说明书2）
PDF_PATHS = [DATA_DIR / "招股说明书1.pdf", DATA_DIR / "招股说明书2.pdf"]

# ---- 多文档实体消歧 ----------------------------------------------------------
# 两份招股说明书都存在「法定代表人 / 注册资本 / 关联方」等同构字段，
# 必须用公司名做文档级路由，否则会出现跨文档串味（问力源却召回兴图）。
DOC_COMPANY = {
    "招股说明书1.pdf": ["武汉兴图新科电子股份有限公司", "兴图新科电子", "兴图新科", "兴图"],
    "招股说明书2.pdf": ["武汉力源信息技术股份有限公司", "力源信息技术", "力源信息", "力源"],
}
DOC_FILTER_ENABLE = os.getenv("DOC_FILTER_ENABLE", "1") == "1"   # 是否启用文档级路由
W_DOC = float(os.getenv("W_DOC", "0.16"))                       # 文档匹配加成权重

# ---- 模型配置（Ollama 本地服务） ---------------------------------------------
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-r1:1.5b")        # 本地大模型
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "bge-m3:567m")  # 本地向量模型
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "16"))    # 批量嵌入大小
EMBED_TIMEOUT = float(os.getenv("EMBED_TIMEOUT", "120"))
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "60"))

# ---- 切片参数 ----------------------------------------------------------------
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "600"))       # 结构感知切片大小（字符）
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))  # 切片重叠
MIN_CHUNK_SIZE = int(os.getenv("MIN_CHUNK_SIZE", "60"))  # 过短片段合并阈值
TABLE_MIN_ROWS = int(os.getenv("TABLE_MIN_ROWS", "2"))   # 表格最少行数（过滤误检）
TABLE_MIN_COLS = int(os.getenv("TABLE_MIN_COLS", "2"))   # 表格最少列数

# ---- 图形语义（沿用 04 工单能力：组织结构图 → 可检索文本） ---------------------
USE_FIGURE_SEMANTICS = os.getenv("USE_FIGURE_SEMANTICS", "1") == "1"  # 是否解析图形语义
FIGURE_MIN_BOXES = int(os.getenv("FIGURE_MIN_BOXES", "4"))   # 图形区域最少节点框数
FIGURE_MAX_BOXES = int(os.getenv("FIGURE_MAX_BOXES", "60"))  # 图形区域最多节点框数

# ---- 检索参数 ----------------------------------------------------------------
TOP_K = int(os.getenv("TOP_K", "5"))                    # 最终返回片段数
VECTOR_TOP_K = int(os.getenv("VECTOR_TOP_K", "30"))     # 向量召回候选数
BM25_TOP_K = int(os.getenv("BM25_TOP_K", "30"))         # BM25 召回候选数
RANK_FUSION_K = int(os.getenv("RANK_FUSION_K", "60"))   # RRF 窗口
ANSWER_POOL = int(os.getenv("ANSWER_POOL", "30"))       # 答案合成候选池大小
ANSWER_TOP_K = int(os.getenv("ANSWER_TOP_K", "12"))     # 参与答案合成的重排片段数

# 重排权重（加权线性融合，权重之和约为 1）
W_VECTOR = float(os.getenv("W_VECTOR", "0.22"))         # 语义相似度
W_BM25 = float(os.getenv("W_BM25", "0.16"))             # 词法命中
W_KEYWORD = float(os.getenv("W_KEYWORD", "0.20"))       # 查询关键词覆盖率
W_NUMERIC = float(os.getenv("W_NUMERIC", "0.10"))       # 数值/年份匹配
W_ENTITY = float(os.getenv("W_ENTITY", "0.08"))         # 实体匹配
W_TABLE = float(os.getenv("W_TABLE", "0.05"))           # 表格块加成
W_FIGURE = float(os.getenv("W_FIGURE", "0.05"))         # 图形块加成（本工单新增）
W_TOPIC = float(os.getenv("W_TOPIC", "0.08"))           # 多轮话题实体加成
W_SYN = float(os.getenv("W_SYN", "0.18"))               # 同义词覆盖加成（本工单新增）
W_SERIES = float(os.getenv("W_SERIES", "0.15"))         # 数值序列信号（分别为/合计分别为）

# ---- 同义词扩展（Query 理解优化：跨表述召回） ---------------------------------
# 同一事实在招股说明书中存在多种表述（如「军用领域」=「国防客户」/「军方市场」），
# 仅靠字面匹配会漏召回；此处为关键业务词建立同义词表，用于检索查询扩展与重排加成。
SYNONYMS = {
    "军用": ["国防", "军方", "军品", "军队"],
    "国防": ["军用", "军方", "军品"],
    "军方": ["军用", "国防"],
    "收入": ["销售额", "营业收入", "销售收入"],
    "销售额": ["收入", "营业收入", "销售收入"],
    "组织结构图": ["组织架构图", "组织结构"],
    "销售处": ["销售网点", "销售办事处"],
}

# ---- 多轮对话（本工单核心） ---------------------------------------------------
# 会话历史保留的最大轮数（超出后最早的轮次被压缩为话题摘要）
MAX_TURNS = int(os.getenv("MAX_TURNS", "8"))
# 触发「需要改写」的信号：出现指代词 / 省略式追问
COREF_PRONOUNS = ["他", "她", "它", "他们", "它们", "其", "该", "此", "这", "那",
                  "这个", "那个", "这家", "那家", "该公司", "上述", "上述公司"]
ELLIPSIS_PATTERNS = [r"^那[^，。？]*呢[？?]?$", r"^呢[？?]?$", r"^还有呢[？?]?$",
                     r"^那么[^，。？]*呢[？?]?$"]
USE_LLM_REWRITE = os.getenv("USE_LLM_REWRITE", "0") == "1"  # 是否用本地 LLM 辅助改写

# ---- 答案生成 ----------------------------------------------------------------
EXTRACTIVE_MAX_SENTENCES = int(os.getenv("EXTRACTIVE_MAX_SENTENCES", "4"))
# 相关性下限：最高重排分低于此值视为「与文档无关」，返回友好兜底而非强行抽取
MIN_ANSWER_SCORE = float(os.getenv("MIN_ANSWER_SCORE", "0.35"))

# ---- 性能目标 ----------------------------------------------------------------
TARGET_RESPONSE_SECONDS = 3.0   # 需求：从提问到生成答案 <= 3s
TARGET_ACCURACY = 0.90          # 需求：问答准确率 >= 90%


def ensure_dirs() -> None:
    """确保关键目录存在（首次运行自动创建）。"""
    for d in (DATA_DIR, DB_DIR, OUTPUT_DIR, FIGURE_OUT_DIR, SHOT_DIR):
        try:
            d.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass


ensure_dirs()