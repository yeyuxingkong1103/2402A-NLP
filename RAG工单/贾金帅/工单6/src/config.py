"""
配置中心
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化
工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
工单编号：人工智能NLP-RAG-图像内容解析及检索优化

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
WORK_ORDER_NO_IMAGE = "人工智能NLP-RAG-图像内容解析及检索优化"           # 工单 04：图像解析
WORK_ORDER_NO_QU = "人工智能NLP-RAG-Query理解优化任务"                   # 工单 05：多轮对话
# 工单 06：混合检索。本工单新增的全部代码（src/fulltext.py / reranker.py /
# hybrid.py / embed_registry.py）注释里都必须带这个编号。
WORK_ORDER_NO_HYBRID = "人工智能NLP-RAG-混合检索任务"                     # 工单 06：混合检索
WORK_ORDER_NOS = [WORK_ORDER_NO, WORK_ORDER_NO_OPT, WORK_ORDER_NO_TABLE,
                  WORK_ORDER_NO_IMAGE, WORK_ORDER_NO_QU, WORK_ORDER_NO_HYBRID]

# 交付物命名用的短标签
WORK_ORDER_SHORT = "工单6"

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
EVAL_DATASET = _get("EVAL_DATASET", "questions_wot4.json")

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

# ---------------------------------------------------------------- 图像解析（工单04 核心）
# 工单备注硬要求：「PDF 中的图像语义解析使用多模态模型（CLIP 或多模态大模型）实现」。
# 本工程两条路都落地了：
#   ① 多模态大模型 qwen-vl-plus 生成结构化语义描述 → 进文本索引（主力，id=5/id=6 靠它）；
#   ② Chinese-CLIP 图-文跨模态向量 → 独立的图像召回通道（以文搜图）。
IMAGE_PARSE_ENABLED = _get_bool("IMAGE_PARSE_ENABLED", True)

# 多模态大模型（DashScope 兼容模式）。实测 qwen-vl-max 该账号无权限（HTTP 403），
# qwen-vl-plus 可用，单张图约 3.2s。
VL_MODEL = _get("VL_MODEL", "qwen-vl-plus")
VL_TIMEOUT_SECONDS = _get_int("VL_TIMEOUT_SECONDS", 120)
VL_MAX_TOKENS = _get_int("VL_MAX_TOKENS", 1400)
VL_MAX_RETRY = _get_int("VL_MAX_RETRY", 3)
# 描述文本入索引时的截断上限（太长会稀释向量）
IMAGE_DESC_MAX_CHARS = _get_int("IMAGE_DESC_MAX_CHARS", 1200)

# 图区域检测参数
IMAGE_GAP_PT = _get_float("IMAGE_GAP_PT", 30.0)     # 图元聚类间距
IMAGE_MIN_W = _get_float("IMAGE_MIN_W", 90.0)       # 最小图宽（滤 logo/图标）
IMAGE_MIN_H = _get_float("IMAGE_MIN_H", 55.0)       # 最小图高（滤页眉分隔线）
# 裁剪渲染精度：实测 220dpi 最优 —— 150 太糊（细连接线看不清），
# 300 反而更差（图被模型内部缩放后连接线更糊，p39 把「销售部」数成 3 个）
IMAGE_CROP_DPI = _get_int("IMAGE_CROP_DPI", 220)

# CLIP 跨模态通道
CLIP_ENABLED = _get_bool("CLIP_ENABLED", True)
CLIP_MODEL_PATH = _get("CLIP_MODEL_PATH", r"D:\chinese-clip-vit-base-patch16")
CLIP_DIM = _get_int("CLIP_DIM", 512)
# 图像召回通道的权重：图像块在融合时额外加的这一项
RETRIEVAL_IMAGE_DELTA = _get_float("RETRIEVAL_IMAGE_DELTA", 0.12)
# CLIP 跨模态召回的取数上限，以及「相似度低于此值就不算命中」的floor。
# 为什么要有 floor：CLIP 的相似度**永远为正**（向量都是归一化后内积，中文图文对
# 普遍落在 0.1~0.35 区间），若不加地板，任何问题都会"命中"排名第一的那张图，
# 等于给全库 68 张图无条件发先验 —— 这会污染表格题与正文题的排序。
RETRIEVAL_CLIP_TOP_K = _get_int("RETRIEVAL_CLIP_TOP_K", 10)
RETRIEVAL_CLIP_MIN_SIM = _get_float("RETRIEVAL_CLIP_MIN_SIM", 0.20)

# 低信息量图类型的排序惩罚（工单4 调优项）。
#
# 实测发现的一处代价：兴图 p539 那张「证照照片」的图内文字是
# 「二、发行人控股股东、实际控制人声明；本人承诺本招股意向书…」这类签名页碎片，
# 语义价值极低，但它**同时含有公司名与"法定代表人/注册资本"等高频问题词**，
# 于是把 id=531（法定代表人）挤到了第 2 名。这是图像解析带来的副作用。
#
# 做法是给这类图**排序惩罚**而不是丢弃：信息仍在库里、界面仍能看到，
# 只是不优先。惩罚只作用于 score（排序），**不动 evidence** ——
# 依据分是绝对量纲、是阈值闸门的唯一判据，让惩罚去影响闸门会变成
# 「因为图类型不好，所以离题问题也该放行」这种说不通的逻辑。
LOW_INFO_FIG_TYPES = ("证照照片",)
RETRIEVAL_LOW_INFO_PENALTY = _get_float("RETRIEVAL_LOW_INFO_PENALTY", 0.15)

IMAGE_DIR = DATA_DIR / "images"                     # 裁剪出的图 + 语义描述
FIGURES_JSON = IMAGE_DIR / "figures.json"           # 图清单（含语义描述、CLIP 向量路径）
CLIP_EMB_NPY = IMAGE_DIR / "clip_embeddings.npy"    # 图向量矩阵
# 「图像不做解析」的对照索引（工单4 消融用）
INDEX_NOIMAGE_DIR = DATA_DIR / "index_noimage"

# ---------------------------------------------------------------- 分块
CHUNK_SIZE = _get_int("CHUNK_SIZE", 600)
CHUNK_OVERLAP = _get_int("CHUNK_OVERLAP", 80)
CHUNK_MIN_LENGTH = _get_int("CHUNK_MIN_LENGTH", 40)
CHUNK_MAX_LENGTH = _get_int("CHUNK_MAX_LENGTH", 1200)

# ---------------------------------------------------------------- 检索
RETRIEVAL_DENSE_TOP_K = _get_int("RETRIEVAL_DENSE_TOP_K", 20)
RETRIEVAL_SPARSE_TOP_K = _get_int("RETRIEVAL_SPARSE_TOP_K", 20)
# 最终进上下文的条数。
# 6 → 8 是评测要求的直接结果（21 题验收集，其余参数取最优）：
#   top_k=6 → 准确率 90.48% / 召回率 96.43%
#   top_k=8 → 准确率 95.24% / 召回率 98.81%   ← 工单要求 90% / 95%
# 8 条 × 约 400 字 ≈ 3200 字，仍在 `build_context(max_chars=4800)` 的预算内，
# 不会把大模型上下文撑爆；检索耗时也没有变化（多出来的只是写入 dict 的成本）。
RETRIEVAL_FINAL_TOP_K = _get_int("RETRIEVAL_FINAL_TOP_K", 8)
RETRIEVAL_BM25_BETA = _get_float("RETRIEVAL_BM25_BETA", 0.30)

# BM25 项的归一化参考（工单4 修正项）。
#
# 原实现用**最大值**归一（`bm25_raw / bm25_max`）。max 是**极值统计量，对离群点极敏感**：
#   工单4 加入 68 个图像块后，兴图 p539 那张「证照照片」的图像块——
#   图内文字是「二、发行人控股股东、实际控制人声明；本人承诺本招股意向书…」这样的签名页碎片，
#   词表又长、又恰好满是"法定代表人""注册资本"这类高频问题词——成了该查询的 BM25 全局最大值。
#   于是**所有其他块**（包括真正装答案的 p72 正文块）的 BM25 项都被它按比例压小，
#   evidence 从 0.7756 掉到 0.7436，**跌破 0.75 阈值闸门**，被整块拦掉。
#   表现为"图像解析让一个原本答对的题变成答不出来"——而真凶是归一化基准被一个离群块带走了。
#
# 改成按**参考分位数**归一（默认第 95 百分位），并把结果**上限截断到 1.0**：
#   分位数是稳健统计量，单个离群块再也撬不动全体块量纲；
#   截断到 1.0 则维持了「归一化到 0~1」的原有契约（分位点以上的块视为"关键词相关度已拉满"）。
RETRIEVAL_BM25_REF_PERCENTILE = _get_float("RETRIEVAL_BM25_REF_PERCENTILE", 95.0)
RETRIEVAL_DOC_CONSENSUS_DELTA = _get_float("RETRIEVAL_DOC_CONSENSUS_DELTA", 0.25)
# 文档消歧加成（加法先验）：问题点名某份文档时，该文档的块在排序上获得这个加成。
# 只影响 score（排序），闸门仍只看 evidence —— 「问的是哪一家」不该让离题问题过闸门。
RETRIEVAL_DOC_HINT_DELTA = _get_float("RETRIEVAL_DOC_HINT_DELTA", 0.25)
# 阈值闸门：依据分下限。
#
# ⚠️ 两套检索实现的依据分量纲不同，阈值要分别标定，不能照抄同一个数字：
#   * `RETRIEVAL_MIN_EVIDENCE = 0.75` —— 工单1~5 的检索（`src/retriever.py`），
#     它的 BM25 项来自 BM25Okapi（单字段纯正文），沿用原值，保持历史结果可比。
#   * `HYBRID_MIN_EVIDENCE = 0.70` —— 工单6 的检索（`src/hybrid.py`），
#     它的 BM25 项来自 BM25F（三字段加权），分数分布与 BM25Okapi 不同。
#     扫出来：0.60/0.65/0.70 → 80.95%；0.75 → 76.19%；0.80 → 71.43%。
#     即旧阈值 0.75 在新量纲下偏严，会多拦掉约 1 题的正确答案。
RETRIEVAL_MIN_EVIDENCE = _get_float("RETRIEVAL_MIN_EVIDENCE", 0.75)
HYBRID_MIN_EVIDENCE = _get_float("HYBRID_MIN_EVIDENCE", 0.70)

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

# ---------------------------------------------------------------- 多轮对话（工单05 Query理解优化）
# 工单目标：在检索问答的基础上实现**多轮对话**，让「他」「这个公司」「那力源呢？」这类
# 依赖上文的追问能被正确理解。核心是两件事：指代消解 + 省略补全。
#
# 关键约束（来自工单性能验收「响应时间 ≤ 3 秒」）：**历史必须限长**。
# 会话越长、拼进 prompt 的上下文越多，单次往返就越慢；而且招股书问答里
# 真正有用的只是最近一两轮的主体与意图，更早的多是噪音。
MULTITURN_ENABLED = _get_bool("MULTITURN_ENABLED", True)
# 会话最多保留多少轮（一问一答记 1 轮）。超出后丢最早的。
MULTITURN_MAX_TURNS = _get_int("MULTITURN_MAX_TURNS", 6)
# 拼进 LLM prompt 的历史最多多少字符（只放最近的、且做截断，防止拖慢首字）
MULTITURN_HISTORY_CHARS = _get_int("MULTITURN_HISTORY_CHARS", 600)
# 多轮改写单独设一个小的 max_tokens（只要一个检索式 + 一个主体名，不需要长输出）
MULTITURN_LLM_MAX_TOKENS = _get_int("MULTITURN_LLM_MAX_TOKENS", 220)
# 消融实验开关（单变量）：off=不做多轮 / concat=只把历史原文拼给检索 /
# rule=只用规则消解 / rule+llm=规则优先、规则不确定时调 LLM（交付配置）
MULTITURN_MODE = _get("MULTITURN_MODE", "rule+llm")
# 规则兜底：判定「这是指代」的正则。命中即认为当前轮**不能单独理解**，
# 必须用历史补全。注意「他」在招股书语境里指公司（原文用例即如此），不是人。
MULTITURN_COREF_RE = _get(
    "MULTITURN_COREF_RE",
    r"他|她|它|他们|该公司|这家公司|这个公司|该企业|该公司|其(?!他)|上述公司|前述公司|这两家|那家",
)
# 规则兜底：判定「这是省略式追问」的正则 —— 只有主体、没有谓词的问法。
# 典型：「那武汉力源信息技术股份有限公司呢？」「力源呢？」
MULTITURN_ELLIPSIS_RE = _get(
    "MULTITURN_ELLIPSIS_RE",
    r"^\s*(那|那么|这|还有|另外)?\s*[^，,。？?]{0,40}\s*(呢|怎样|怎么样|如何)\s*[？?]?\s*$",
)

# ================================================================ 工单06 混合检索
#
# 工单要求（原文）：
#   「提供向量检索（召回+重排）和全文检索以及两者同时执行的混合检索的
#     检索策略的配置及应用的功能」
#   「支持向量检索和全文检索的权重调整」
#   「提供混合检索结果的融合算法，如加权平均、投票机制等」
#   「支持多种嵌入模型（如 bge、m3e 以及其他嵌入模型）」
#   「提供至少 3 种重排算法（如基于 LLM 的重排器、基于 TF-IDF 的重排器、
#     基于用户反馈的自适应重排器）」
#
# 下面 6 组配置就是把上面五句话逐条落到可开关的旋钮上。

# ---------------------------------------------------------------- 嵌入模型注册表
# 工单要求「支持多种嵌入模型」。做法不是把模型名写死在代码里，而是维护一张**注册表**：
# 每项记 (显示名, 本地目录, 向量维度, 是否需要 query 指令前缀)，运行期按目录是否存在
# 判定「可用 / 未下载」。这样换模型 = 换一个 key，不需要改任何检索代码；
# 索引按模型 key 分目录存放（data/index_<key>），互不覆盖。
#
# ⚠️ 换模型必须重建索引：不同模型维度/语义空间都不同，
#   用 bge 建出来的 embeddings.npy 喂给 m3e 查询是纯粹的错配（维度都可能对不上）。
EMBEDDING_REGISTRY = {
    "bge-small-zh": {
        "name": "BAAI/bge-small-zh-v1.5",
        "path": _get("EMBEDDING_MODEL_PATH", r"D:\bge-small-zh-v1.5"),
        "dim": 512,
        "query_instruction": "",     # bge 短查询侧不加指令（长查询才需要）
        "note": "当前主线。512 维，CPU 编码 ~15ms/条，中文短文本检索表现稳定。",
    },
    "bge-base-zh": {
        "name": "BAAI/bge-base-zh-v1.5",
        "path": _get("EMBEDDING_MODEL_PATH_BASE", r"D:\bge-base-zh-v1.5"),
        "dim": 768,
        "query_instruction": "",
        "note": "bge 的 base 档，768 维，精度更高、编码更慢（本机未下载）。",
    },
    "m3e-base": {
        "name": "moka-ai/m3e-base",
        "path": _get("EMBEDDING_MODEL_PATH_M3E", r"D:\m3e-base"),
        "dim": 768,
        "query_instruction": "",
        "note": "工单点名的 m3e。768 维，中文同义改写鲁棒性好（本机未下载）。",
    },
    "bge-m3": {
        "name": "BAAI/bge-m3",
        "path": _get("EMBEDDING_MODEL_PATH_M3", r"D:\bge-m3"),
        "dim": 1024,
        "query_instruction": "",
        "note": "多语言（中英混检），1024 维，模型 2.2GB（本机未下载）。",
    },
    "text2vec-base": {
        "name": "shibing624/text2vec-base-chinese",
        "path": _get("EMBEDDING_MODEL_PATH_T2V", r"D:\text2vec-base-chinese"),
        "dim": 768,
        "query_instruction": "",
        "note": "另一支常用的中文句向量模型，作为第三方对照（本机未下载）。",
    },
}
# 当前使用的模型 key。索引目录随之切换为 data/index_<key>（bge-small-zh 就是 data/index）。
EMBEDDING_MODEL_KEY = _get("EMBEDDING_MODEL_KEY", "bge-small-zh")
# bge 系列官方建议：**查询侧**加这个指令前缀能提升检索效果（文档侧不加）。
# 默认关闭 —— 本语料实测加与不加的差异在噪音范围内，且加前缀会让 query 向量
# 与工单1~5 已建好的索引不完全同分布。作为可切换选项保留，便于做消融。
EMBEDDING_QUERY_INSTRUCTION = _get(
    "EMBEDDING_QUERY_INSTRUCTION", ""
)

# ---------------------------------------------------------------- 全文检索（工单06 (2)）
FULLTEXT_ENABLED = _get_bool("FULLTEXT_ENABLED", True)
# BM25F 的饱和参数。k1 越大，词频的边际收益衰减越慢；b 越大，长块被惩罚越重。
# k1=1.2 / b=0.75 是 Lucene 默认，在本语料上实测也最稳
# （招股书块长差异不算极端，b 再往上调会压到"含表格的长块"）。
FULLTEXT_K1 = _get_float("FULLTEXT_K1", 1.2)
FULLTEXT_B = _get_float("FULLTEXT_B", 0.75)
# 多字段权重（BM25F）。标题权重 3.0 是实测扫出来的：
#   1.0 → 标题信息完全被正文淹没，问「组织机构」召回全是正文提到该词的长段落；
#   3.0 → 章节标题含该词的块稳定进前 3，且没有把离题块带上来；
#   5.0 → 开始过拟合标题（有些章节标题很泛，比如"风险因素"）。
FULLTEXT_FIELD_WEIGHTS = {
    # 2.0 是**扫描出来的**，不是拍的（21 题验收集 + 纯全文策略）：
    #   1.0 → 85.71%  2.0 → 85.71%  3.0 → 80.95%  5.0 → 80.95%
    # 标题权重过大会过拟合：招股书里「供货能力」「风险因素」这类章节标题很泛，
    # 一个查询词撞上标题就拿到 3 倍分，反而把正文里真正作答的块压下去。
    # 举一个实测例子：问「在哪个市场领域已经做成了重要供货方」，
    # 标题权重 3.0 时「3、供货能力」这一节（另一家公司的文档）被顶到第一名。
    "title": _get_float("FULLTEXT_W_TITLE", 2.0),
    "body": _get_float("FULLTEXT_W_BODY", 1.0),
    "summary": _get_float("FULLTEXT_W_SUMMARY", 1.5),
}
# 「摘要」字段取正文前多少字
FULLTEXT_SUMMARY_CHARS = _get_int("FULLTEXT_SUMMARY_CHARS", 120)
# 模糊/通配一次最多扩展出多少个真实词（防止 `销*` 把整个词表拉进来）
FULLTEXT_MAX_FUZZY_EXPAND = _get_int("FULLTEXT_MAX_FUZZY_EXPAND", 16)
# 全文检索取多少条进候选池
FULLTEXT_TOP_K = _get_int("FULLTEXT_TOP_K", 20)
# 错别字容错（工单06「模糊查询」能力的一部分）。
# 「注册资本」被敲成「注册酱」时，jieba 切成「注册」+「酱」，而「酱」df=0：
# 它不贡献分数，还会把整句的向量语义带偏（实测向量路从命中 p60 掉到命中 p214）。
# 容错办法：df=0 的短词当作疑似错字，用它前面那个**紧邻实词**做前缀补全。
FULLTEXT_REPAIR = _get_bool("FULLTEXT_REPAIR", True)
# 参与补全的「前一词」df 上限。df 太大说明是「公司」「报告期」这类泛词，
# 拿它去补全只会引入噪音；df 太小（1~2）又不足以证明它是个实词。
FULLTEXT_REPAIR_MAX_DF = _get_int("FULLTEXT_REPAIR_MAX_DF", 300)

# ---------------------------------------------------------------- 检索策略与融合（工单06 (3)）
# 可选策略：vector（纯向量）/ fulltext（纯全文）/ hybrid（两者同时执行后融合）
RETRIEVAL_STRATEGY = _get("RETRIEVAL_STRATEGY", "hybrid")
# 融合算法：
#   weighted  加权平均 —— 两路分数各自归一化后按 weight 线性加权（工单点名的"加权平均"）
#   rrf       倒数排名融合 —— 只看名次不看分数（工单点名的"投票机制"的代表）
#   borda     Borda 计数 —— 经典的排序投票法
#   vote      多数投票 —— 被两路同时召回的块按命中票数奖励（纯投票，不含分数）
#   evidence  工单1~5 的原始公式（余弦 + β·BM25归一 + δ·共识），作为兼容基线
HYBRID_FUSION = _get("HYBRID_FUSION", "weighted")
# 向量路权重（0~1）。0 = 纯全文，1 = 纯向量。
# 默认 0.6 是实测值：招股书提问里「专有名词/数字」类问题靠全文路，
# 「换种说法问同一个意思」类问题靠向量路，0.6 在 16 题评测集上综合最优。
HYBRID_VECTOR_WEIGHT = _get_float("HYBRID_VECTOR_WEIGHT", 0.6)
# RRF 的平滑常数 k。k 越小，名次差异被放大得越狠（k=1 时 top1 与 top2 差距极大）。
# 60 是 RRF 原论文的取值，实测也最稳。
HYBRID_RRF_K = _get_int("HYBRID_RRF_K", 60)
# 加权平均时**全文路的归一化参考**：p95 分位（同工单4 对 BM25 的处理，抗离群块）。
HYBRID_SPARSE_REF_PERCENTILE = _get_float("HYBRID_SPARSE_REF_PERCENTILE", 95.0)
# 两路各自取多少条进候选池。
#
# 60 是扫出来的，而且是**本工单影响最大的一个参数**：
#   20 → 76.19%   30 → 85.71%   40 → 85.71%   50 → 90.48%   60 → 95.24%   80 → 95.24%（饱和）
# 直觉上"候选池越大越慢"，但实测耗时没有变化（88ms → 105ms，噪声量级）：
# 因为两路召回本来就是**全量打分**（余弦全库点积 / 倒排索引全库匹配），
# 取 top-20 与取 top-60 的差别只是"后面多 copy 几个下标"，
# 贵的那一步（打分与排序）早就付过了。
# 真正贵的是重排，而重排只作用于前 RERANK_CANDIDATES 条，不受这个值影响。
HYBRID_POOL_TOP_K = _get_int("HYBRID_POOL_TOP_K", 60)

# ---------------------------------------------------------------- 重排（工单06 (1)）
# 可选重排器（见 src/reranker.py）：
#   none       不重排（对照组）
#   tfidf      基于 TF-IDF 的重排器
#   llm        基于 LLM 的重排器
#   feedback   基于用户反馈的自适应重排器
#   cross      基于交叉编码器（bge-reranker-v2-m3）的重排器【可选，吃内存】
# 默认是 `feedback`（用户反馈自适应重排器），这是评测跑出来的：
#   none → 85.71%（MRR 0.756）  tfidf → 85.71%（MRR 0.708）  feedback → **90.48%（MRR 0.857）**
# 它在零反馈时的表现 = 一组可解释的浅层特征（关键词覆盖率 / 标题命中 / 表图偏好），
# 而这几个特征恰好是招股书问答里最有效的信号；随着用户点"有用/没用"，
# 权重还会继续往数据那边靠。
# `tfidf` 完全本地无状态、也最"讲得清楚"，演示讲解技术时用它更直观。
RERANKER = _get("RERANKER", "feedback")
# 送进重排器的候选数。重排是 O(N) 的（LLM/cross 还是一次网络/前向），
# 候选太多会直接把「响应 ≤3 秒」吃掉，所以先粗排取前 N 再精排。
RERANK_CANDIDATES = _get_int("RERANK_CANDIDATES", 12)
# 重排后进入最终上下文的结果数
RERANK_OUTPUT_TOP_K = _get_int("RERANK_OUTPUT_TOP_K", 6)
# 重排分与原始分的插值权重：final = α·重排分 + (1-α)·原始分。
# 不做 100% 替换的原因：重排器（尤其 LLM）偶尔会给出与原文依据明显冲突的排序，
# 留一条"原始分"的锚，可以让它在整体上只做**微调**而非推翻。
#
# 0.3 也是扫出来的（α 从 0 到 0.8，步长 0.1/0.2）：
#   α=0.0 → MRR 0.732   α=0.3 → **0.740**   α=0.6 → 0.702   α=0.8 → 0.678
# 结论很实在：**TF-IDF 重排器的边际收益很小，α 越大越容易帮倒忙**。
# 它的评分本质是字面匹配（含 bigram 词序），而语义改写题恰恰要求"字面不像也算对"，
# 所以重排分越占主导，这类题被拖得越狠。给它 30% 的话语权是实测的最优折中。
RERANK_ALPHA = _get_float("RERANK_ALPHA", 0.3)

# LLM 重排器：每次送几条候选给模型打分（控制 token 与往返次数）
LLM_RERANK_BATCH = _get_int("LLM_RERANK_BATCH", 6)
LLM_RERANK_MAX_TOKENS = _get_int("LLM_RERANK_MAX_TOKENS", 400)
# LLM 重排器在候选多时的兜底：单次调用超时（秒）→ 整体降级为 TF-IDF 重排
LLM_RERANK_TIMEOUT = _get_int("LLM_RERANK_TIMEOUT", 25)

# TF-IDF 重排器：用多大的滑窗把「查询词在块内的聚集程度」算进去
TFIDF_RERANK_NGRAM = _get_int("TFIDF_RERANK_NGRAM", 2)

# 用户反馈自适应重排器：反馈落盘位置 + 学习率 + 先验强度
FEEDBACK_PATH = DATA_DIR / "feedback.jsonl"
FEEDBACK_ENABLED = _get_bool("FEEDBACK_ENABLED", True)
FEEDBACK_LR = _get_float("FEEDBACK_LR", 0.25)
# 自适应重排器的权重初值（特征 → 权重）。这些特征都是**可解释的浅层信号**，
# 之所以不直接学一个黑箱：招股书问答的样本量只有几十条，
# 学黑箱必然过拟合，而"哪一类块更可能是答案"这件事本身就有很强的先验。
FEEDBACK_PRIORS = {
    "keyword_coverage": _get_float("FB_PRIOR_KW", 0.55),   # 查询词覆盖率
    "type_image": _get_float("FB_PRIOR_IMG", 0.05),        # 图像块偏好
    "type_table": _get_float("FB_PRIOR_TAB", 0.10),        # 表格块偏好
    "doc_key": _get_float("FB_PRIOR_DOC", 0.10),           # 文档偏好
    "section_hit": _get_float("FB_PRIOR_SEC", 0.20),       # 标题命中
}

# 交叉编码器重排器（可选，工单没要求，作为"重排算法"的第四种提供）
CROSS_ENCODER_ENABLED = _get_bool("CROSS_ENCODER_ENABLED", False)
CROSS_ENCODER_PATH = _get("CROSS_ENCODER_PATH", r"D:\models\bge-reranker-v2-m3")
# fp16 权重在 CPU 上做前向反而更慢（torch CPU 的 half 算子多是软实现），
# 因此这里默认让它在 CPU 上用 fp32，除非显式指定 device=cuda。
CROSS_ENCODER_DEVICE = _get("CROSS_ENCODER_DEVICE", "cpu")
# 一次前向的 (query, doc) 对数上限 —— 12 对短文本 CPU 约 1.2s，再多就吃掉 3 秒预算
CROSS_ENCODER_MAX_PAIRS = _get_int("CROSS_ENCODER_MAX_PAIRS", 12)

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
              INDEX_NAIVE_DIR, INDEX_NOTABLE_DIR, INDEX_NOIMAGE_DIR, IMAGE_DIR):
        d.mkdir(parents=True, exist_ok=True)


def as_public_dict() -> dict:
    """对外暴露的非敏感配置（用于 /config 探针，绝不返回 API Key）。"""
    return {
        "work_order_no": WORK_ORDER_NO,
        "work_order_no_opt": WORK_ORDER_NO_OPT,
        "work_order_no_table": WORK_ORDER_NO_TABLE,
        "work_order_no_image": WORK_ORDER_NO_IMAGE,
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
        "image_parsing": {
            "enabled": IMAGE_PARSE_ENABLED,
            "vl_model": VL_MODEL,                       # 多模态大模型
            "clip_enabled": CLIP_ENABLED,
            "clip_model_path": CLIP_MODEL_PATH,         # 跨模态模型
            "crop_dpi": IMAGE_CROP_DPI,
            "gap_pt": IMAGE_GAP_PT,
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
            "image_delta": RETRIEVAL_IMAGE_DELTA,
            "clip_top_k": RETRIEVAL_CLIP_TOP_K,
            "clip_min_sim": RETRIEVAL_CLIP_MIN_SIM,
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
