# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化
# 工单编号：人工智能NLP-RAG-混合检索任务
# 工单02 - 基于PDF文档的问答系统的优化
# 工单06 - 混合检索（检索模式 / 融合 / 重排器）
"""
检索剖面（profile）：把「用哪套检索策略」变成可切换的配置对象。

【为什么要做剖面，而不是直接改参数】
工单02 的产出物是「对比优化前后检索精确度的变化」。要出这个对比，
就必须能**在同一份向量库、同一套 prompt 上**反复切换检索策略跑评测 ——
否则比出来的是"两个不同系统"，不是"优化前后的差异"。

【为什么是 frozen dataclass + 显式传参，而不是改全局 settings】
`settings` 是 `@lru_cache` 单例（见 config.py），进程内只有一份。
SSE 流式接口 + `asyncio.Semaphore` 排队下，若按请求去改写全局 settings，
两个并发请求会互相踩到对方刚改的参数（竞态），而且极难复现。
所以剖面是不可变值对象，沿调用链显式传下去；`settings.retrieval_profile`
只提供"默认用哪个"。

【三个剖面分别是谁】
- baseline   ：重构出来的**朴素地板**（单路召回、无重排、无冗余过滤、300 字窗口）。
               它不是工单01 的交付物，是用来量化"每项优化贡献多少"的下界参考。
- delivered  ：**工单01 的交付版本**，也就是工单02 诚实的「优化前」状态。
               它的参数必须与工单01 完全一致，是重构的回归闸门。
- optimized  ：工单02 的优化版本（同事实聚合 + 权威页优先 + 邻块扩展）。

【注意 baseline 与 delivered 不是一回事】
写文档时要说清：`delivered` 才是"优化前"，`baseline` 是消融对照的地板。
否则 `baseline` 的差数字会被误读成"工单01 退步了"。
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from app.config import WORK_ORDER_ID_02, settings

# 工单02 的工单编号（与工单01 的不同，见各自工单 PDF 的「备注」栏）
OPT_WORK_ORDER_ID = WORK_ORDER_ID_02

# 上下文窗口的硬下限。
# 【为什么钉死 600】工单01 踩过这个坑：窗口设 300 字时，id=793 召回的片段
# （1-1-151，408 字）里含答案的那句「下游行业为各类终端用户……主要包括军队、
# 政府机关、能源」在 336 字处，被窗口切掉，模型只看到前文图注里的
# 「军队企事业单位」，答案就缺了政府机关与能源 —— **答案不是检索错了，是被
# 截断毁了**。这类失败是静默的（接口照常返回、不报错），所以要用断言钉死，
# 而不是靠注释提醒。
# baseline 是唯一豁免：它存在的意义就是复现"优化前"的坏行为。
MIN_CONTEXT_CHARS = 600


@dataclass(frozen=True)
class RetrievalProfile:
    """一套检索策略。不可变 —— 改动请用 `replace()` 派生新实例。"""

    name: str

    # ---------- 召回 ----------
    # 双路 = 原问句 + 抽象版（去掉公司全称）。单路只走原问句。
    dual_view: bool = True
    pool_size: int = 30          # 每路召回的候选池大小

    # ---------- 重排 / 过滤 ----------
    lexical_rerank: bool = True  # dense + 词法 线性融合
    w_dense: float = 0.5
    diverse: bool = True         # select_diverse：整块 3-gram Jaccard 去字面重复
    dup_jaccard: float = 0.85

    # ---------- 工单02 新增：同事实聚合 ----------
    # 招股书同一事实在十几页重复，且各页表述互相矛盾（见 retriever.cluster_facts）。
    # 整块相似度抓不到它们（实测 Jaccard 仅 0.055），必须下沉到句子粒度。
    aggregate_facts: bool = False
    authority_bias: bool = False
    fact_ngram: int = 6
    fact_sim: float = 0.65       # 句子级 n-gram 包含率阈值（已标定，见 retriever）
    fact_merge_chars: int = 0    # 代表块之外可并入的补充句总字数；0 = 只用代表块

    # ---------- 工单02 新增：邻块扩展（small-to-big 的近似） ----------
    expand_neighbors: bool = False
    neighbor_radius: int = 1
    neighbor_chars: int = 400        # 单个邻块上限（走 best_window，不整块塞）
    neighbor_total_chars: int = 900  # 全部邻块合计上限

    # ---------- 上下文 ----------
    top_k: int = 3
    context_chunk_chars: int = 600

    # ==================================================================
    # 工单06：检索模式 / 融合 / 重排器
    # ==================================================================
    # 【全部带默认值，且默认值 = 工单05 的行为】三个既有剖面一个新字段都不写，
    # 靠"默认值 + retriever 里的显式分流"保证路径逐字节不变（16/16 回归红线）。
    retrieval_mode: str = "vector"      # vector | fulltext | hybrid
    fusion: str = "rrf"                 # rrf（投票） | weighted（加权平均）；仅 hybrid
    fusion_impl: str = "app"            # app（应用层融合） | milvus（向量库原生 Ranker）
    w_keyword: float = 0.5              # hybrid 下关键词路的权重（0~1）
    keyword_pool_size: int = 30         # 全文检索那一路的候选池

    reranker: str = "lexical"           # lexical | tfidf | llm | feedback | none
    rerank_top_n: int = 5               # LLM 重排只打前 N 个（控 TTFT）
    rerank_timeout_ms: int = 900        # 超时即回退 lexical，并留痕

    # 反馈重排器的闸门：样本太少时**必须不生效**，否则会把偶然偏好放大成排序
    feedback_min_support: int = 2
    feedback_max_boost: float = 0.15

    def __post_init__(self) -> None:
        # 见 MIN_CONTEXT_CHARS 的说明：除了 baseline，谁都不许把窗口降到 600 以下
        if self.name != "baseline" and self.context_chunk_chars < MIN_CONTEXT_CHARS:
            raise ValueError(
                f"剖面 {self.name!r} 的 context_chunk_chars="
                f"{self.context_chunk_chars} 低于硬下限 {MIN_CONTEXT_CHARS}。"
                f"这会静默切掉答案（工单01 的 id=793 就是这么坏的）。"
                f"只有 baseline 允许，且必须在报告里标注它已知会截断答案。"
            )
        # ---------- 工单06：枚举与范围全部**当场报错**，不静默回退 ----------
        # 【为什么不静默回退】写错一个字母（比如 mode="hybird"）若被悄悄换成默认值，
        # 评测出来的"混合检索没有提升"其实根本没跑混合检索 —— 这是最贵的静默失败。
        if self.retrieval_mode not in ("vector", "fulltext", "hybrid"):
            raise ValueError(f"未知 retrieval_mode={self.retrieval_mode!r}"
                             f"（可选：vector / fulltext / hybrid）")
        if self.fusion not in ("rrf", "weighted"):
            raise ValueError(f"未知 fusion={self.fusion!r}（可选：rrf / weighted）")
        if self.fusion_impl not in ("app", "milvus"):
            raise ValueError(f"未知 fusion_impl={self.fusion_impl!r}（可选：app / milvus）")
        if self.fusion_impl == "milvus" and self.retrieval_mode != "hybrid":
            raise ValueError("fusion_impl=milvus 只在 retrieval_mode='hybrid' 下有意义")
        if self.reranker not in ("lexical", "tfidf", "llm", "feedback", "none"):
            raise ValueError(f"未知 reranker={self.reranker!r}"
                             f"（可选：lexical / tfidf / llm / feedback / none）")
        if not (0.0 <= self.w_keyword <= 1.0):
            raise ValueError(f"w_keyword 必须在 [0,1]，实得 {self.w_keyword}")
        if self.rerank_top_n < 1:
            raise ValueError(f"rerank_top_n 必须 >= 1，实得 {self.rerank_top_n}")
        if self.reranker == "llm" and self.rerank_timeout_ms < 100:
            raise ValueError("rerank_timeout_ms 太小（<100ms），LLM 重排不可能完成")

    # ------------------------------------------------------------------
    def derived(self, **kw) -> "RetrievalProfile":
        """派生一个改了几个字段的新剖面（扫描调参用）。"""
        return replace(self, **kw)


# ----------------------------------------------------------------------
# 三个剖面
# ----------------------------------------------------------------------
PROFILES: dict[str, RetrievalProfile] = {
    # 优化前的地板：单路召回、纯 dense 打分、不做冗余过滤、300 字窗口。
    # 刻意保留这些"坏"参数，用来演示每一项优化的边际贡献。
    "baseline": RetrievalProfile(
        name="baseline",
        dual_view=False,
        pool_size=10,
        lexical_rerank=False,
        diverse=False,
        top_k=3,
        context_chunk_chars=300,
    ),
    # 工单01 交付版 —— 也是工单02 对比里的「优化前」。
    # 这里的默认值必须与工单01 完全一致：它是重构的回归闸门。
    "delivered": RetrievalProfile(name="delivered"),
    # 工单02 优化版。
    #
    # 【这些参数是扫参扫出来的结论，不是拍脑袋】scripts/sweep.py 跑了
    # top_k{3,4,5} × pool{20,30,50} × 窗口{600,700} × 邻块{开,关} × 邻字{450,900}
    # 共 72 格，结论是：
    #   - `page_recall` 在 72 格中**恒为 0.567**，这几个旋钮根本推不动它；
    #   - top_k 3→5 让 CKC 停在 0.967 封顶不涨，上下文却从 1647 涨到 2793、
    #     页精确率反从 0.300 掉到 0.180；
    #   - pool 开到 50 反而让 CKC 掉到 0.867（池子越大 → 聚类合并越多 →
    #     代表块可能丢掉某个关键词）；
    #   - 阶段二真实生成里**每一格都是 10/10**，连上下文 2931 字的对照格也是，
    #     且 TTFT-P95 仍只有 2118ms。
    # 所以 top_k=3 / pool=30 / 600 字就是最优点，保持默认即可。
    # 邻块扩展保持开启：实测平均只增加 44 字（1691 vs 1647），
    # 因为互补性过滤把冗余邻块几乎全挡掉了，代价可忽略。
    "optimized": RetrievalProfile(
        name="optimized",
        aggregate_facts=True,
        authority_bias=True,
        expand_neighbors=True,
    ),
}

DEFAULT_PROFILE = "delivered"


# ----------------------------------------------------------------------
# 工单06 的新剖面
# ----------------------------------------------------------------------
# 【为什么不直接塞进 PROFILES】`scripts/eval.py --compare all` 会遍历 PROFILES
# 逐个跑完整评测（含生成与 LLM judge）。往里加 5 个 = 5 倍时间与模型调用。
# 所以核心三剖面留在 PROFILES（`--compare all` 的语义与耗时都不变），
# 工单06 的剖面单独一张表，靠 `get_profile` / `profile_names` 统一解析。
EXTRA_PROFILES: dict[str, RetrievalProfile] = {
    # 纯全文检索（不起向量、不嵌入）：布尔/短语/模糊/多字段全部可用。
    # 排序就是 BM25F 本身（reranker="none"）—— 那才是"全文检索"的原貌，
    # 叠上查询词覆盖率重排就说不清是谁在起作用了。
    "fulltext": RetrievalProfile(
        name="fulltext", retrieval_mode="fulltext",
        lexical_rerank=False, reranker="none",
    ),
    # 混合检索（默认 RRF 投票融合）：向量路 ⊕ 全文路
    # 【为什么保留覆盖率高重排（reranker="lexical"）】实测这才是关键：
    # 只做融合、不做覆盖率重排时，页召回与 CKC 都明显低于向量基线；
    # 保留覆盖率重排后，同等上下文规模下「答案要点召回率」比纯向量高 6~10 个点
    # （top-3 84.4%→91.7%，top-8 86.9%→96.9%）。融合负责"把关键词路的好块拉进候选"，
    # 覆盖率重排负责"在候选里挑出真正含答案要点的"。两者缺一不可。
    # 【为什么带上工单02 的三件套】实测对比发现：optimized 在评测里补了 **32 个邻块**，
    # 而最初的 hybrid 剖面没开邻块扩展 —— 恰好 id=2（引子句与表格被切在相邻两块）
    # 与 id=5（图块与枚举）的答案就在邻块里，于是混合版的端到端掉到 14/16。
    # 混合检索是**在工单02 的基础上再加一路**，不是另起一套更弱的链路。
    # 【交付配置：混合 + 邻块扩展】实测（同一批 16 题、同一份库）：
    #   规则命中 16/16（=100%）｜要点召回（含邻块）100%｜CKC 93.8%（向量基线 89.2%）
    #   ｜TTFT-P50 1925ms、≤3s 14/16
    # 邻块扩展**必须开**：id=2（引子句与表格被切在相邻两块）与 id=5（图块枚举）
    # 的答案就在邻块里，关掉会掉到 14/16。
    "hybrid": RetrievalProfile(
        name="hybrid", retrieval_mode="hybrid", fusion="rrf", w_keyword=0.5,
        lexical_rerank=False, reranker="lexical", expand_neighbors=True,
    ),
    # 混合 + 工单02 全部三件套（含同事实聚合）：CKC 略低、且 cluster_facts 在更大的
    # 候选集上是 O(n²) 句子比对，实测把 ≤3s 从 14/16 拖到 13/16。保留作对照。
    "hybrid_full": RetrievalProfile(
        name="hybrid_full", retrieval_mode="hybrid", fusion="rrf", w_keyword=0.5,
        lexical_rerank=False, reranker="lexical",
        aggregate_facts=True, authority_bias=True, expand_neighbors=True,
    ),
    # 不带三件套的裸混合（消融对照：看"混合"本身贡献了多少）
    "hybrid_bare": RetrievalProfile(
        name="hybrid_bare", retrieval_mode="hybrid", fusion="rrf", w_keyword=0.5,
        lexical_rerank=False, reranker="lexical",
    ),
    # 加权平均融合（工单点名的第二种融合算法，作为对照）
    "hybrid_weighted": RetrievalProfile(
        name="hybrid_weighted", retrieval_mode="hybrid", fusion="weighted",
        w_keyword=0.3, lexical_rerank=False, reranker="lexical", expand_neighbors=True,
    ),
    # 向量库原生混合检索（Milvus sparse + WeightedRanker / RRFRanker），工程化对照
    "hybrid_milvus": RetrievalProfile(
        name="hybrid_milvus", retrieval_mode="hybrid", fusion_impl="milvus",
        lexical_rerank=False, reranker="lexical",
    ),
    # 混合 + 工单02 的三件套（同事实聚合 / 权威页优先 / 邻块扩展）+ 上下文放宽到 5 块
    "optimized_hybrid": RetrievalProfile(
        name="optimized_hybrid", retrieval_mode="hybrid", fusion="rrf", w_keyword=0.5,
        lexical_rerank=False, reranker="lexical", top_k=5,
        aggregate_facts=True, authority_bias=True, expand_neighbors=True,
    ),
    # LLM 重排（演示/离线用：它要吃一次生成调用，见文档里的 TTFT 披露）
    "hybrid_llm": RetrievalProfile(
        name="hybrid_llm", retrieval_mode="hybrid", fusion="rrf", w_keyword=0.5,
        lexical_rerank=False, reranker="llm",
    ),
    # 反馈重排（依赖工单05 的 feedback.jsonl；冷启动时等价于不重排）
    "hybrid_feedback": RetrievalProfile(
        name="hybrid_feedback", retrieval_mode="hybrid", fusion="rrf", w_keyword=0.5,
        lexical_rerank=False, reranker="feedback",
    ),
    # TF-IDF 重排（工单点名的第二种重排算法；实测在本语料上弱于覆盖率重排，
    # 保留为对照，不作默认）
    "hybrid_tfidf": RetrievalProfile(
        name="hybrid_tfidf", retrieval_mode="hybrid", fusion="rrf", w_keyword=0.5,
        lexical_rerank=False, reranker="tfidf",
    ),
}

ALL_PROFILES: dict[str, RetrievalProfile] = {**PROFILES, **EXTRA_PROFILES}


def get_profile(name: str | None = None) -> RetrievalProfile:
    """按名字取剖面；未知名回退到默认剖面（不抛异常，避免接口 500）。"""
    key = (name or settings.retrieval_profile or DEFAULT_PROFILE).strip().lower()
    return ALL_PROFILES.get(key, ALL_PROFILES[DEFAULT_PROFILE])


def profile_names() -> list[str]:
    """已注册的剖面名（前端下拉、脚本参数校验用）。"""
    return list(ALL_PROFILES)


def core_profile_names() -> list[str]:
    """工单01-05 的核心三剖面（`--compare all` 用，避免把耗时乘 5）。"""
    return list(PROFILES)
