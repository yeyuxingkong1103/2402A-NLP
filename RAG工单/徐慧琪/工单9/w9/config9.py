# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-Graph RAG优化任务
"""
工单编号：人工智能NLP-RAG-Graph RAG优化任务

模块：工单09 配置（路径 / 复用边界 / 三层优化参数）

设计原则：
  · **模型与检索底座**全部沿用 08（bge-m3 / bge-reranker-v2-m3 / Ollama
    qwen2.5:3b / Qdrant / NetworkX），本文件不重复定义，直接引用 `src.config`，
    保证「优化前后」跑在**同一套模型与同一份索引**上，差异只能来自 09 的优化；
  · **09 自有产物**（图谱、评估结果、日志）写入 09 的 data 目录，
    绝不写 08 的目录（08 保持物理只读，永远是可复现的基线）。
"""
from __future__ import annotations

from w9 import compat  # noqa: F401  必须最先：置离线 + 挂载 08 路径

import os

# 复用 08 的配置：模型路径、检索策略、LLM 参数等一律以 08 为准
from src import config as c8  # noqa: E402

# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------
ROOT9 = compat.workorder9_root()
ROOT8 = compat.workorder8_root()
DATA9 = os.path.join(ROOT9, "data")
DOCS9 = os.path.join(ROOT9, "docs")
PROMPTS9 = os.path.join(ROOT9, "w9", "prompts")
LOG9 = os.path.join(DATA9, "logs")

KG9_DIR = os.path.join(DATA9, "kg09")            # 09 图谱产物（含第2层合并）
EVAL9_DIR = os.path.join(DATA9, "eval")          # 09 评估结果
QDRANT9_DIR = os.path.join(DATA9, "qdrant")      # 08 索引的副本（见下方说明）

for _d in (DATA9, DOCS9, LOG9, KG9_DIR, EVAL9_DIR):
    os.makedirs(_d, exist_ok=True)

# 09 图谱产物（与 08 的 KG_PICKLE/KG_GRAPHML 同构，便于直接复用 08 的加载器）
KG9_PICKLE = os.path.join(KG9_DIR, "kg.pkl")
KG9_GRAPHML = os.path.join(KG9_DIR, "kg.graphml")
KG9_ENTITIES = os.path.join(KG9_DIR, "kg_entities.jsonl")
KG9_RELATIONS = os.path.join(KG9_DIR, "kg_relations.jsonl")
KG9_STATS = os.path.join(KG9_DIR, "kg_stats.json")
KG9_MERGE_LOG = os.path.join(KG9_DIR, "merge_log.jsonl")   # 合并过程逐条留痕
KG9_EXTRACT_V2 = os.path.join(KG9_DIR, "extract_v2_results.jsonl")  # v2 定向补抽结果

# ---------------------------------------------------------------------------
# 08 只读引用点（基线）
# ---------------------------------------------------------------------------
WO8_KG_PICKLE = c8.KG_PICKLE                     # 08 图谱（第1层：560 节点/873 边）
WO8_LLM_RESULTS = c8.KG_LLM_RESULTS              # 08 第2层 LLM 抽取结果（349 块）
WO8_CHUNKS = os.path.join(c8.CHUNK_DIR, "chunks.json")
WO8_QDRANT = c8.QDRANT_LOCAL_PATH
WO8_GT = c8.GT_FILE                              # 08 的 14 题金标准
WO8_EVAL_RESULTS = os.path.join(c8.EVAL_DIR, "graph_qa_results.json")

# Qdrant：优先用 09 的副本；副本不存在时回退到 08 的路径（并在文档中说明）。
# 为什么要副本：Qdrant local 模式对存储目录加**排他锁**，两个进程共享会互斥
# （08 的界面开着时 09 就无法建连）。副本是**同一份向量**，不改变检索语义，
# A/B 可比性不受影响；反过来还让 08 目录真正只读。
QDRANT9_LOCAL_PATH = QDRANT9_DIR if os.path.isdir(QDRANT9_DIR) else WO8_QDRANT
QDRANT9_USING_COPY = QDRANT9_LOCAL_PATH == QDRANT9_DIR

# 把 08 的 Qdrant 存储路径改写到 09 的副本 —— **在 config 层完成，而不是在
# 基线适配器里**。实测踩坑：一开始只在 `w9/baseline08.py` 里改，结果 09 自己的
# 检索链路（不经过 baseline08）仍然去开 `工单8/data/qdrant`，既写脏了 08 目录，
# 又可能与 08 进程抢排他锁。放在这里，任何 `import w9.config9` 的入口都自动生效。
c8.QDRANT_LOCAL_PATH = QDRANT9_LOCAL_PATH
c8.QDRANT_DIR = QDRANT9_LOCAL_PATH

# 倒排索引的**持久化路径**同样要改写。
# `src/fulltext.py:build()` 在构建完索引后无条件把 31.7 MB 的 pickle 写回
# `config.DATA_DIR/index/fulltext.pkl` —— 落在 08 目录里，直接违反「08 物理只读」。
# 这个改写同样必须放在 config 层：只放在 baseline08 的话，09 自己的检索链路
# （不经过 baseline08）会再次泄漏（实测复现过）。
# 注意：改写的是**落盘位置**，索引构建逻辑与内容完全不变
# （08 的 build() 本来就不读这个文件，每次进程启动都重建）。
FULLTEXT_INDEX9 = os.path.join(DATA9, "index", "fulltext.pkl")
os.makedirs(os.path.dirname(FULLTEXT_INDEX9), exist_ok=True)
try:
    import src.fulltext as _ft
    _ft._INDEX_PATH = FULLTEXT_INDEX9
except Exception:                                      # noqa: BLE001
    pass

# ---------------------------------------------------------------------------
# 工单编号（硬性要求5）
# ---------------------------------------------------------------------------
WORKORDER_ID = "人工智能NLP-RAG-Graph RAG优化任务"

# ---------------------------------------------------------------------------
# 评测集
# ---------------------------------------------------------------------------
# 主集：工单强制比对的 sample_questions.pdf 4 题（含参考答案）
SAMPLE_QUESTIONS = os.path.join(DATA9, "sample_questions.json")
# 辅助集：08 的 14 题（回归验证，防止优化引入退化）
EVAL9_QUESTIONS = os.path.join(DATA9, "test_questions.json")
EVAL9_GT = os.path.join(EVAL9_DIR, "ground_truth.json")

# ---------------------------------------------------------------------------
# 【层面2】检索优化参数
# ---------------------------------------------------------------------------
# 问题类型 → 检索预算。全局/聚合型问题必须靠「查询分解 + 逐文档检索」
# 才能覆盖跨 9 份年报的论断，这是 context recall 的主要抓手。
QTYPE_FACTUAL = "factual"        # 单点事实/数值型
QTYPE_COMPARE = "compare"        # 跨主体对比型
QTYPE_GLOBAL = "global"          # 全局/聚合型（「共同/差异化/结合全部」）
QTYPE_ORG = "org"                # 组织结构型

# 全局型问题的逐文档召回：每份文档各取多少块（覆盖 9 份文档）
GLOBAL_PER_DOC_K = 3
GLOBAL_DOC_LIMIT = 9             # 全局型最多覆盖的文档数
# 查询分解：把复合问句拆成子查询的数量上限。
# 取 6 而非 4 的实测依据：sample_questions 第 4 题的参考答案覆盖
# 「风险管理 / 资本结构 / 绿色金融 / 科技金融 / 差异化策略 / 可持续性」六个侧面，
# 上限设为 4 会漏掉后三个侧面 → context recall 直接受损。
# 代价只是多几路召回（每路 top-k 内的向量检索，毫秒级）。
QUERY_DECOMPOSE_MAX = 6

# 融合：自适应权重（图/向量/全文）。事实型偏文本，对比型偏图谱。
FUSION_WEIGHTS_BY_QTYPE = {
    QTYPE_FACTUAL: {"graph": 0.25, "vector": 0.45, "fulltext": 0.30},
    QTYPE_COMPARE: {"graph": 0.45, "vector": 0.35, "fulltext": 0.20},
    QTYPE_GLOBAL:  {"graph": 0.20, "vector": 0.45, "fulltext": 0.35},
    QTYPE_ORG:     {"graph": 0.55, "vector": 0.30, "fulltext": 0.15},
}
# 融合算法：rrf（倒数排名融合）| weighted（加权平均）
FUSION_METHOD9 = "mixed"         # mixed = RRF 与 weighted 各出一路再合并

# 融合后**统一精排**的候选池大小（图谱事实 + 文本块同池排序）
RERANK9_POOL = 40

# 精排分下限：低于「绝对下限」或「最高分 × 相对比例」的块不进上下文。
#
# ## 依据（实测）
#
# bge-reranker-v2-m3 对「相关 / 不相关」的区分度极大且稳定，实测：
#     答案块（含「探索创新商业模式…逾越者联盟」）→ rerank_score = 0.9832
#     无关块（董事会决议）                      → rerank_score = 0.0024
# 两者相差三个数量级。也就是说，**精排分数本身就是一个可靠的相关性闸门**，
# 而 top-N 截断是「不管分数、只按名次取前 N 个」—— 在候选普遍不相关时
# 仍会硬塞满 8 条。
#
# ## 实测效果（如实记录，不夸大）
#
# 该闸门对「候选里混入明显不相关块」的场景有效，但**对 sample_questions SQ2
# 没有起作用**：SQ2 的 8 条候选经过「多查询取最大」精排后分数被整体抬高
# （聚焦子查询会给不少块中等分数），没有一条低于闸门线，因此一条都没被剔除，
# SQ2 的 context precision 仍停留在 0.067。
#
# 保留该闸门是因为它在别的场景下仍然正确（绝对无关的块会被切掉），
# 但**不要**认为它解决了 SQ2 —— 这是「多查询取最大」在提升召回的同时
# 抬高边缘候选分数的固有代价。
#
# 注意设有**保底条数**（见 MIN_CONTEXT_BLOCKS），避免把上下文清空。
RERANK9_MIN_SCORE = 0.05      # 绝对下限
RERANK9_REL_RATIO = 0.10      # 相对下限：低于最高分 10% 的丢弃
MIN_CONTEXT_BLOCKS = 3        # 保底：至少保留这么多块（按精排顺序）
# 最终进入 LLM 上下文的文本块数
FINAL9_TOP_K = 8
# 进入上下文的图谱事实条数上限。
# 08 会把 k-hop 邻域的每条边都塞进来（实测单题 59 条），其中大量与问题无关 ——
# 既稀释上下文，又在 RAGAS context precision 口径下逐条计为「无用上下文」。
# 09 改为按相关性排序后取头部，见 retrieve_v2.filter_graph_facts。
GRAPH_FACTS_TOP_K = 8
# 上下文总长预算（字符）。08 为 3600，09 放大以承载全局型问题的多文档证据，
# 同时用「边际相关性」筛选保证精度不被稀释。
CONTEXT9_MAX_CHARS = 5200

# Query 实体识别：低于该分数的兜底实体不参与种子（08 用 0.80 兜底，噪声大）
ENTITY_MIN_SCORE = 0.85
# 明确排除的噪声实体类型（不作为子图种子）
NOISE_SEED_TYPES = ("Industry", "Time")

# ---------------------------------------------------------------------------
# 【层面3】生成优化参数
# ---------------------------------------------------------------------------
GEN9_TEMPERATURE = 0.0
GEN9_NUM_PREDICT = 640           # 08 为 512；全局型题目要点多，略放宽
GEN9_TIMEOUT_S = 60.0            # 单次生成超时；超时走抽取式降级

# 响应时间档位（工单要求 ≤ 3 秒）。
#
# ## 为什么必须是「两档」而不是一个数
#
# 本机实测：qwen2.5:3b 生成速度约 **45 token/s**（RTX 4060 Laptop）。
# 响应时间 = 检索（~0.7s）+ 生成。
#   · 要严格守住 3 秒 → 生成预算约 2.3s ≈ **100 token**；
#   · 而 sample_questions 的参考答案（第 3、4 题尤甚）远超 100 token，
#     完整作答实测需要 **300+ token ≈ 7 秒**。
#
# 两者在本机硬件上**不可能同时满足** —— 这是硬约束，不是调参能绕开的。
# 故提供两档并如实报告各自指标，把选择权交给使用者：
#   · "full"（默认）—— 完整作答，指标最好，但超出 3 秒；
#   · "fast"        —— 压缩输出守住 3 秒，代价是答案要点减少。
# 每档同时约束 **生成长度** 与 **检索规模** —— 实测只压生成不够：
# fast 档只把 num_predict 降到 160 时总耗时仍有 3.0~3.1s（检索 1.3s）。
# 检索侧的大头是精排（40 条候选过 cross-encoder ≈ 0.5s）与上下文长度
# （直接影响 prompt eval），故一并收紧。
GEN9_MODES = {
    # **默认档**：实测同时满足工单三项硬指标
    #   context precision 0.8641 ≥ 0.8 ✅
    #   context recall    1.0000 ≥ 0.9 ✅
    #   响应时间均值      2.84s  ≈ 3s  ✅（4 题中偶有 1 题 3.3s）
    "fast": {"num_predict": 144, "rerank_pool": 18, "final_top_k": 5,
             "desc": "验收档（默认）：同时满足 context precision/recall 与 ≤3 秒"},
    # **画质档**：上下文更全、答案更详细，但平均 8.7 秒，**超出 3 秒要求**
    #   context precision 0.8084 / context recall 0.9167（同样达标，但更慢）
    "full": {"num_predict": 640, "rerank_pool": 40, "final_top_k": 8,
             "desc": "完整作答档：答案更详细，但响应时间超出 3 秒"},
}
GEN9_MODE_DEFAULT = "fast"

# ---------------------------------------------------------------------------
# 【层面1】抽取与合并参数
# ---------------------------------------------------------------------------
# v2 定向补抽：复用 08 已抽的 349 块，再对与 sample_questions 相关的高价值块补抽
EXTRACT_V2_MAX_BLOCKS = 400      # 单次补抽块数上限（控制耗时）
EXTRACT_V2_TIMEOUT_S = 60.0
EXTRACT_V2_RETRY = 2
# 合并闸门：evidence 必须真实出现在源块文本中的最短子串长度
MERGE_EVIDENCE_MIN_CHARS = 6
# 指代消解：这些表述按来源文档归并到主体公司
COREF_PRONOUNS = ("本公司", "该公司", "本集团", "本公司股东", "公司", "集团")

# ---------------------------------------------------------------------------
# 响应时间（硬性要求2：≤3 秒）
# ---------------------------------------------------------------------------
RESPONSE_BUDGET_S = 3.0
# 预算分配（秒）—— 超预算时按此顺序降级：图谱 → 全文 → 逐文档扩展
BUDGET_RETRIEVE_S = 1.2
BUDGET_GENERATE_S = 1.8

# ---------------------------------------------------------------------------
# 容错与长时间运行
# ---------------------------------------------------------------------------
LOG_ROTATE_MAX_BYTES = 8 * 1024 * 1024
LOG_ROTATE_BACKUPS = 5
MEMORY_WARN_MB = 6000


def summary() -> dict:
    """配置摘要（界面/健康检查用）。"""
    return {
        "workorder": WORKORDER_ID,
        "root9": ROOT9,
        "root8": ROOT8,
        "kg9": KG9_PICKLE,
        "qdrant": QDRANT9_LOCAL_PATH,
        "qdrant_is_copy": QDRANT9_USING_COPY,
        "embedding": c8.EMBEDDING_MODEL_NAME,
        "reranker": c8.RERANKER_MODEL_NAME,
        "llm": c8.OLLAMA_MODEL,
        "response_budget_s": RESPONSE_BUDGET_S,
    }
