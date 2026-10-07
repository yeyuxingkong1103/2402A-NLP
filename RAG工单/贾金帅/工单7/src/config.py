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
WORK_ORDER_NOS = [WORK_ORDER_NO, WORK_ORDER_NO_OPT,
                  WORK_ORDER_NO_TABLE, WORK_ORDER_NO_IMAGE, WORK_ORDER_NO_QU]

# 交付物命名用的短标签
WORK_ORDER_SHORT = "工单5"

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
RETRIEVAL_FINAL_TOP_K = _get_int("RETRIEVAL_FINAL_TOP_K", 6)
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
# 多轮确定主体后，是否**硬排除**另一份文档（工单05 补充调优）。
# 判据来自实测：力源法定代表人那道题，兴图新科占了 5/8 个上下文名额且都过了闸门，
# 把正确答案挤出去了。多轮层既然已经知道问的是哪一家，另一家就是纯噪音。
# 只在多轮生效（ MULTITURN_MODE != "off" ），单轮行为与工单1~4 完全一致。
MULTITURN_DOC_FILTER_ENABLED = _get_bool("MULTITURN_DOC_FILTER_ENABLED", True)

# ---- 主体一致性排序主键（工单05 补充调优，doc_filter 之后的第二道）----
# 只把「另一份文档」排掉还不够：**同一份文档里也有主体不符的块**。
# 实测（力源法定代表人）：问题问「武汉力源信息技术股份有限公司的法定代表人」，
# 排在第一的是 p26「(六)申请上市证券交易所：深圳证券交易所 法定代表人：宋丽萍」——
# 它字面完美匹配「法定代表人」，BM25 归一后满分 1.0，但它答的是**深交所**的法定代表人。
# 而真正的权威块 p23「一、发行人的基本情况 …4、法定代表人：赵马克」排第 5、ev=0.7219。
# 稠密向量也救不了：p26 通篇在讲「法定代表人：宋丽萍」，语义上确实"更像"。
# 唯一确定性的判据是**主体一致性** —— 问「X 的法定代表人」，答案所在块必然同时
# 出现 X 和「法定代表人」。而这一点只有多轮层知道（主体可能是继承来的）。
# 做法：把「块文本是否含本轮主体名」当作**排序主键**（先分组、组内再按分排序），
# 而不是加一个可调权重 —— 因为「主体对不对」是硬条件，不是可以权衡的偏好。
MULTITURN_SUBJECT_FIRST_ENABLED = _get_bool("MULTITURN_SUBJECT_FIRST_ENABLED", True)

# 主体已锁定时放宽的绝对闸门（见 RETRIEVAL_MIN_EVIDENCE 的原始用途）。
# 0.75 这道闸门是工单2 为**多文档通用场景**设的「离题防线」：问题是哪家公司的都不确定，
# 只能靠绝对分数判断「这段内容到底沾不沾边」。而多轮层已经把主体锁到唯一一份文档、
# 并且要求块里出现主体名 —— 离题风险已被这两道前置条件消除，此时再拿通用闸门去砍，
# 砍掉的是**真正相关但表述偏长/偏散**的权威块（p23 的 0.7219 就是这么被砍掉的）。
# 取 0.70：足以让 p23 这种「含主体名 + 含答案」的块进榜，同时仍能挡住明显离题的块。
MULTITURN_RELAXED_EVIDENCE = _get_float("MULTITURN_RELAXED_EVIDENCE", 0.70)


# 规则消解结果的试检索「可信线」：低于它就认为规则没搞定，需要 LLM 兜底
MULTITURN_RULE_MIN_EVIDENCE = _get_float("MULTITURN_RULE_MIN_EVIDENCE", 0.75)
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
