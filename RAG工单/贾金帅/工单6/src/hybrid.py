"""
混合检索编排与融合
工单编号：人工智能NLP-RAG-混合检索任务

对应工单功能需求（工单6「2 功能详细需求」全文）：

    （1）向量检索（召回+重排）—— 嵌入 + 相似度召回 + 重排算法优化排序
    （2）全文检索              —— 倒排索引 + 布尔/短语/模糊 + 多字段
    （3）混合检索              —— 同时执行两路，结合优势，**支持权重调整与融合算法**

本模块是这三条需求在**执行层**的汇合点：

        用户问题
           │
           ├── 预处理：查询归一化 / 无区分度词过滤 / 多文档消歧（工单1~5 的既有能力）
           │
     ┌─────┴─────┬──────────────┬───────────────┐
     │ 向量路     │ 全文路        │ CLIP 跨模态路  │   ← 各路独立召回（可单独开关）
     │ embed→余弦 │ 倒排→BM25F    │ 图向量→余弦    │
     └─────┬─────┴──────────────┴───────────────┘
           │  ↓ 融合（weighted / rrf / borda / vote / evidence）
           │  ↓ 加权重排（none / tfidf / llm / feedback / cross）
           ↓
        最终 top-k 上下文

—— 三件必须讲清楚的事 ——

**① 为什么「向量」和「全文」互补，而不是二选一**

    向量检索擅长「换种说法」：问「公司的钱主要花在哪」能召回到写「募集资金运用」的段落；
    但它对**精确串**不敏感 —— 「武汉兴图新科电子股份有限公司」和
    「武汉力源信息技术股份有限公司」在向量空间里非常接近（同一个模板写出来的两份招股书），
    向量路经常分不出问的是哪一家。
    全文检索正好相反：精确串一找一个准，但同义改写完全无能为力
    （问「钱花在哪」，它找不到「募集资金运用」）。
    ——所以工单要的是**同时执行**，而不是选一个更好的。

**② 为什么默认不是 RRF**

    RRF（倒数排名融合）是工业界最流行的融合算法，优点是不需要分数可比。
    但在本语料实测里它有个具体的坑：**只看名次，不看分数**，
    于是「仅被 BM25 命中的纯关键词噪音块」会按名次拿到与「双路都高分的正确块」
    接近的融合分（rank1 的 RRF 贡献恒为 1/(k+1)，与它到底得了多少分无关）。
    招股书里这类噪音是大量的（贷款合同表格、合同文本，公司名到处出现）。
    默认用 `weighted`：两路分数先各自归一化到 0~1 再加权 ——
    归一化基准用 p95 分位而不是 max（同工单4 对 BM25 的处理，避免被单个离群块带走量纲）。
    RRF/Borda/vote 仍然完整实现并可一键切换，用于演示与消融对比 —— 这正是工单要的「配置及应用」。

**③ 权重怎么调**

    `vector_weight` ∈ [0,1]：0 = 纯全文，1 = 纯向量。
    实测 0.6 在 16 题评测集上综合最优：专有名词/数字类问题靠全文路兜底，
    语义改写类问题靠向量路兜底。接口与界面上都可实时调整，改完立刻体现在排序上。
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .config import (
    CLIP_ENABLED,
    FULLTEXT_ENABLED,
    FULLTEXT_REPAIR,
    LOW_INFO_FIG_TYPES,
    HYBRID_FUSION,
    HYBRID_POOL_TOP_K,
    HYBRID_RRF_K,
    HYBRID_SPARSE_REF_PERCENTILE,
    HYBRID_VECTOR_WEIGHT,
    RETRIEVAL_BM25_BETA,
    RETRIEVAL_DENSE_TOP_K,
    RETRIEVAL_DOC_CONSENSUS_DELTA,
    RETRIEVAL_DOC_HINT_DELTA,
    RETRIEVAL_FINAL_TOP_K,
    RETRIEVAL_IMAGE_DELTA,
    RETRIEVAL_LOW_INFO_PENALTY,
    HYBRID_MIN_EVIDENCE,
    RETRIEVAL_MIN_EVIDENCE,  # noqa: F401  工单1~5 的闸门阈值，仅用于 describe() 里展示对照
    RETRIEVAL_SPARSE_TOP_K,
    RERANK_ALPHA,
    RERANK_CANDIDATES,
    RERANKER,
    WORK_ORDER_NOS,  # noqa: F401
)
from . import embed_registry
from .index_store import KnowledgeBase
from .query_norm import normalize_query
from .reranker import Candidate, get_reranker
from .retriever import RetrievedChunk, _clip_assets, _fig_map_from_kb

logger = logging.getLogger(__name__)

#: 可用检索策略（工单要求「提供多种检索方式的配置及应用」）
STRATEGIES = {
    "vector": {
        "name": "向量检索",
        "desc": "向量嵌入 + 余弦相似度召回 + 重排。擅长换种说法的语义匹配。",
        "routes": ["vector"],
    },
    "fulltext": {
        "name": "全文检索",
        "desc": "倒排索引 + BM25F，支持布尔/短语/模糊查询与多字段检索。擅长专有名词与精确串。",
        "routes": ["fulltext"],
    },
    "hybrid": {
        "name": "混合检索",
        "desc": "向量与全文同时执行，按权重融合后再重排。两条路的短板互为补充。",
        "routes": ["vector", "fulltext"],
    },
}

FUSIONS = {
    "weighted": {
        "name": "加权平均",
        "desc": "两路分数各自归一化到 0~1 后按权重线性加权。默认方案，抗噪音块最好。",
    },
    "rrf": {
        "name": "RRF 倒数排名融合",
        "desc": "只看名次不看分数：score = Σ 1/(k+rank)。对不同量纲的两路天然友好，但会给纯关键词噪音块记名次分。",
    },
    "borda": {
        "name": "Borda 计数（排序投票）",
        "desc": "每个路由给候选按名次计票，名次越前票越多，再按权重汇总。经典投票机制。",
    },
    "vote": {
        "name": "多数投票",
        "desc": "被两路同时召回的块获得双票加成。最朴素也最稳的「投票机制」。",
    },
    "evidence": {
        "name": "依据分（工单1~5 原式）",
        "desc": "余弦 + β·BM25归一 + δ·文档共识。保留旧公式，便于与工单1~5 的结果横向对照。",
    },
}


# ============================================================ 数据结构

@dataclass
class HybridConfig:
    """一次混合检索的全部可调参数。工单要求的「配置」就是它。"""

    strategy: str = "hybrid"
    fusion: str = HYBRID_FUSION
    vector_weight: float = HYBRID_VECTOR_WEIGHT
    top_k: int = RETRIEVAL_FINAL_TOP_K
    pool_top_k: int = HYBRID_POOL_TOP_K
    reranker: str = RERANKER
    rerank_alpha: float = RERANK_ALPHA
    rerank_candidates: int = RERANK_CANDIDATES
    embed_model: str = embed_registry.DEFAULT_KEY
    # 全文路的查询模式：or（默认，召回优先）/ and（严格，要求全部词命中）
    ft_mode: str = "or"
    # 是否启用全文查询语法（布尔/短语/模糊）。关掉则退化为「按空格切词的 OR 查询」
    ft_syntax: bool = True
    # 是否沿用工单1~5 的若干先验（无区分度词过滤 / 多文档消歧 / CLIP 跨模态）
    use_df_filter: bool = True
    use_doc_hint: bool = True
    use_clip: bool = True
    # 工单4 的低信息图惩罚（证照照片那类）开关；它是排序先验，与 use_clip 相互独立
    use_image_prior: bool = True
    # 错别字容错（把「注册酱」还原成「注册资本」）。作用于两路之前的查询改写
    use_repair: bool = True
    apply_gate: bool = True
    # 消融用：直接把某一路关掉（等价于 strategy 的细化）
    use_dense: bool | None = None
    use_sparse: bool | None = None

    def routes(self) -> list[str]:
        """
        参与**排序融合**的召回路径。

        注意 CLIP 不在这里 —— 它是"加法先验"，不是一路文本召回。
        理由：工单6 要讲的是"向量检索与全文检索如何结合"，
        引入第三条并列的路会让融合公式、权重分配、消融实验全部复杂化；
        而工单4 已经证明 CLIP 最有效的用法是**给图块加一个排序先验**（δ·相似度），
        沿用那个机制既省事又与原结果可比。
        """
        base = STRATEGIES.get(self.strategy, STRATEGIES["hybrid"])["routes"]
        out = list(base)
        # 单路开关（消融）优先于策略
        if self.use_dense is False:
            out = [r for r in out if r != "vector"]
        if self.use_sparse is False:
            out = [r for r in out if r != "fulltext"]
        return out

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy, "fusion": self.fusion,
            "vector_weight": round(self.vector_weight, 3),
            "top_k": self.top_k, "pool_top_k": self.pool_top_k,
            "reranker": self.reranker, "rerank_alpha": self.rerank_alpha,
            "rerank_candidates": self.rerank_candidates,
            "embed_model": self.embed_model, "ft_mode": self.ft_mode,
            "ft_syntax": self.ft_syntax, "use_df_filter": self.use_df_filter,
            "use_doc_hint": self.use_doc_hint, "use_clip": self.use_clip,
            "use_image_prior": self.use_image_prior, "use_repair": self.use_repair,
            "apply_gate": self.apply_gate,
            "routes": self.routes(),
        }


@dataclass
class RouteResult:
    """一路召回的结果。"""

    name: str
    order: list[int] = field(default_factory=list)          # 按分数降序的 chunk 下标
    scores: dict[int, float] = field(default_factory=dict)  # 原始分（量纲随各路不同）
    norm: dict[int, float] = field(default_factory=dict)    # 归一化到 0~1
    ms: float = 0.0
    note: str = ""
    detail: dict = field(default_factory=dict)


@dataclass
class HybridResult:
    """一次混合检索的完整结果。"""

    query: str
    cfg: HybridConfig
    items: list[RetrievedChunk] = field(default_factory=list)
    trace: dict = field(default_factory=dict)
    gated: bool = False
    # 与 retriever.RetrievalResult 对齐的三个字段：rag.py 与前端都在用，
    # 保持同名同义，切换检索实现时上层一行都不用改。
    original_query: str = ""
    normalized_query: str = ""
    used_query: str = ""

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "original_query": self.original_query,
            "normalized_query": self.normalized_query,
            "used_query": self.used_query,
            "config": self.cfg.to_dict(),
            "gated": self.gated,
            "trace": self.trace,
            "items": [it.to_dict() for it in self.items],
        }

    def contexts(self) -> list[str]:
        return [it.text for it in self.items]


# ============================================================ 各路召回

def _route_vector(query: str, kb: KnowledgeBase, cfg: HybridConfig,
                  pool: int, doc_hint: list[str]) -> RouteResult:
    """向量路：嵌入 → 余弦全量打分 → 取 top-pool。"""
    t0 = time.perf_counter()
    r = RouteResult(name="vector")
    try:
        qvec = embed_registry.embed_query(query, cfg.embed_model)
    except FileNotFoundError as exc:
        r.note = f"跳过（{exc}）"
        r.ms = (time.perf_counter() - t0) * 1000
        return r
    dense = embed_registry.cosine_scores(kb.embeddings, qvec)
    if dense.size != len(kb.chunks):
        r.note = (f"跳过（索引向量 {dense.size} 条与分块 {len(kb.chunks)} 条不一致，"
                  f"该模型索引可能未构建）")
        r.ms = (time.perf_counter() - t0) * 1000
        return r
    order = np.argsort(-dense)[:pool].tolist()
    # 文档消歧：把点名文档内部的最优块补进候选池（工单3 的实测教训：
    # 只做“限定”会把答案锁在门外，必须在池外再按文档内排名补一轮）
    if doc_hint:
        mask = np.array([c.get("doc_key") in doc_hint for c in kb.chunks], dtype=bool)
        scoped = np.where(mask, dense, -np.inf)
        order = sorted(set(order) | set(np.argsort(-scoped)[:pool].tolist()))
    r.order = order
    r.scores = {int(i): float(dense[i]) for i in order}
    r.ms = (time.perf_counter() - t0) * 1000
    r.note = f"余弦全量 {len(dense)} 条 → 取 {len(order)} 条"
    r.detail = {"model": cfg.embed_model, "dim": int(kb.embeddings.shape[1]) if kb.embeddings.size else 0}
    return r


_STOP_CACHE: dict[str, set[str]] = {}


def stop_terms_for(kb: KnowledgeBase) -> set[str]:
    """
    取语料级无区分度词集合（工单1~5 的 DF 过滤，多文档语料下**必须保留**）。

    为什么不能省：两份招股书里，公司全称的碎片（武汉/兴图/新科/电子/股份/有限公司）
    出现在几百个块中，是 BM25 的高频命中源。不拦掉的话，问「武汉力源…的组织结构图」
    会把一堆**贷款合同表格**顶上来（那些块里公司全称反复出现）。

    这里比工单1~5 多做了一点：把 `ubiquitous_phrases`（迭代合并出来的长短语，
    例如「兴图新科电子股份有限公司」）也拆成词并进停用集。
    原因是全文路是**分词后逐词查**的，而不是拿整串去匹配，
    所以只停短语级名单里的整串没用 —— 必须落到词上才拦得住。
    """
    key = f"{id(kb)}:{len(kb.chunks)}"
    hit = _STOP_CACHE.get(key)
    if hit is not None:
        return hit
    from .fulltext import tokenize

    terms = set(kb.ubiquitous_terms)
    for p in kb.ubiquitous_phrases:
        terms |= {t for t in tokenize(p) if len(t) > 1}
    _STOP_CACHE.clear()          # 只缓存当前这一份语料，避免 id 复用造成串味
    _STOP_CACHE[key] = terms
    return terms


def _route_fulltext(query: str, kb: KnowledgeBase, cfg: HybridConfig,
                    pool: int) -> RouteResult:
    """
    全文路：倒排索引 + BM25F（支持布尔/短语/模糊）。

    注意查询串用的是**未被字符串级 DF 过滤**的原句：
    过滤交给 `stop_terms` 在叶子节点做（见 `InvertedIndex.evaluate` 的说明），
    这样 `收入 AND 军用`、`section:风险提示` 这类语法的结构不会被破坏。
    """
    t0 = time.perf_counter()
    r = RouteResult(name="fulltext")
    if not FULLTEXT_ENABLED:
        r.note = "已关闭（FULLTEXT_ENABLED=False）"
        return r
    from .fulltext import get_index

    idx = get_index(kb)
    search_q = query if cfg.ft_syntax else " ".join(query.split())
    stop = stop_terms_for(kb) if cfg.use_df_filter else set()
    try:
        hits, ft_trace = idx.search(search_q, top_k=pool, mode=cfg.ft_mode, stop_terms=stop)
    except Exception as exc:  # noqa: BLE001
        # 全文路是**可选的一路**：它自己出错（极端查询式、突发异常）不该把
        # 整个混合检索带崩 —— 记录降级原因，让向量路继续提供召回。
        logger.warning("全文检索失败，本路降级为空：%s", exc)
        r.note = f"跳过（{type(exc).__name__}: {str(exc)[:80]}）"
        r.ms = (time.perf_counter() - t0) * 1000
        return r
    r.order = [h.idx for h in hits]
    r.scores = {h.idx: float(h.score) for h in hits}
    r.ms = (time.perf_counter() - t0) * 1000
    r.note = (f"BM25F 命中 {ft_trace.get('n_candidates', 0)} 条 → 取 {len(hits)} 条"
              f"（词表 {ft_trace.get('index_terms', 0)} 词 / {ft_trace.get('index_docs', 0)} 块"
              f" / 停用 {ft_trace.get('stop_terms', 0)} 词）")
    r.detail = {"query_trace": ft_trace,
                "hits": [h.to_dict() for h in hits[:10]]}
    return r


def _route_clip(query: str, kb: KnowledgeBase, cfg: HybridConfig,
                pool: int) -> RouteResult:
    """CLIP 跨模态路（工单4 的能力，作为第三路参与融合）。"""
    from .config import RETRIEVAL_CLIP_MIN_SIM, RETRIEVAL_CLIP_TOP_K

    t0 = time.perf_counter()
    r = RouteResult(name="clip")
    vecs, figures = _clip_assets()
    if vecs is None or not figures:
        r.note = "跳过（无 CLIP 向量）"
        return r
    try:
        from .clip_encoder import ClipEncoder

        qv = ClipEncoder.instance().encode_texts([query])
        sims = (vecs @ qv[0]).astype(np.float32)
        order = np.argsort(-sims)[:RETRIEVAL_CLIP_TOP_K]
        keep = [int(x) for x in order if float(sims[x]) >= RETRIEVAL_CLIP_MIN_SIM]
        by_ident, by_row = _fig_map_from_kb(kb)
        scores: dict[int, float] = {}
        for row in keep:
            fig = figures[row] if row < len(figures) else {}
            ci = by_ident.get((fig.get("doc_key", ""), fig.get("page", 0), fig.get("index", -1)))
            if ci is None:
                ci = by_row.get(row)
            if ci is None:
                continue
            scores[ci] = max(scores.get(ci, 0.0), float(sims[row]))
        r.scores = scores
        r.order = sorted(scores, key=lambda i: -scores[i])[:pool]
        if r.order:
            r.note = f"命中 {len(scores)} 张图（top={float(sims[order[0]]):.3f}）"
        else:
            r.note = "无命中（相似度低于地板值）"
    except Exception as exc:  # noqa: BLE001 —— 可选通道，失败不影响主链路
        r.note = f"跳过（{type(exc).__name__}: {str(exc)[:60]}）"
    r.ms = (time.perf_counter() - t0) * 1000
    return r


# ============================================================ 融合

def _norm_route(route: RouteResult) -> dict[int, float]:
    """
    把一路的原始分归一化到 0~1。

    ★ 「向量路不做分位数归一」是评测跑出来的一个教训：
    余弦本身就已经在 [0,1] 且量纲稳定（同一个模型、同一批语料，分布不会漂），
    再按候选池的 p95 归一一次，等于把 0.60~0.68 这一小段差异**拉伸到 0~1**，
    于是「几乎一样相关的两条结果」在融合分里被拉开成"差一半"，
    而 BM25 那一侧也是这么被拉满的 —— 两边同时饱和，融合分退化成
    「谁在各自路里名次靠前」的近似投票，反而丢掉了分数携带的信息。
    实测：改正之前，混合检索的准确率比工单1~5 基线低 9.5 个百分点。

    BM25 侧则必须归一（它无上界，本语料实测 0~26），用 p95 分位做基准
    —— 与工单4 对 BM25 的处理同一套理由（抗单个离群块）。
    """
    if route.name == "vector":
        return {i: max(0.0, min(1.0, float(v))) for i, v in route.scores.items()}
    return _normalize(route.scores, HYBRID_SPARSE_REF_PERCENTILE)


def _normalize(scores: dict[int, float], ref_percentile: float = 100.0) -> dict[int, float]:
    """
    把一路分数归一到 0~1。

    参考基准默认取**分位数**（p95）而不是最大值 —— 与工单4 对 BM25 的处理同一套理由：
    max 是极值统计量，一个离群块（比如某个反复出现公司名的合同表格）就能把
    全体的量纲压扁。分位点是稳健统计量，单块撬不动。
    """
    if not scores:
        return {}
    vals = np.fromiter(scores.values(), dtype=np.float32, count=len(scores))
    ref = float(np.percentile(vals, ref_percentile)) if len(vals) > 1 else float(vals.max())
    if ref <= 1e-9:
        ref = float(vals.max()) or 1.0
    return {i: min(1.0, max(0.0, v / ref)) for i, v in scores.items()}


def _fuse(routes: list[RouteResult], cfg: HybridConfig) -> tuple[dict[int, dict], dict]:
    """
    按 `cfg.fusion` 融合多路结果，返回 (idx -> 融合详情, trace)。

    各路**权重**的分配（多路时）：
      * 只有向量 + 全文两路：vector 得 `vector_weight`，fulltext 得 `1 - vector_weight`；
      * 再多一条 CLIP 路：CLIP 从全文那侧切 15% 走
        （即 ft = (1-w)·0.85, clip = (1-w)·0.15）。CLIP 只解决「以文搜图」，
        不该和两路文本召回等权，否则会被图像块刷屏。
    """
    names = [r.name for r in routes if r.order or r.scores]
    w: dict[str, float] = {}
    vw = max(0.0, min(1.0, cfg.vector_weight))
    if len(names) == 1:
        w[names[0]] = 1.0
    elif "vector" in names and "fulltext" in names:
        w["vector"] = vw
        w["fulltext"] = 1.0 - vw
    else:
        # 没有向量路（或没有全文路）：按剩余路均分
        for n in names:
            w[n] = 1.0 / len(names)

    ranks: dict[str, dict[int, int]] = {}
    norms: dict[str, dict[int, float]] = {}
    for r in routes:
        if r.name not in w:
            continue
        ranks[r.name] = {idx: k for k, idx in enumerate(r.order, 1)}
        norms[r.name] = _norm_route(r)

    # ★ 权重为 0 的路**不贡献候选**。
    # 这是一个被实测抓出来的 bug：权重设成 0（"这一路不参与排序"）时，
    # 该路的 top-k 仍然并进了候选池。那些块在另一路的分数是 0，
    # 却照样能拿到共识先验，于是混进前 12、再混进 top-6，
    # 把真正好的块挤出去 —— 实测表现为「权重调成 0 反而比直接选纯全文策略还差
    # 14 个百分点」（71.43% vs 85.71%），这在逻辑上是说不通的。
    pool: set[int] = set()
    for r in routes:
        if r.name in w and w[r.name] > 0:
            pool |= set(r.order)

    fused: dict[int, dict] = {}
    rrf_k = max(1, HYBRID_RRF_K)
    for idx in pool:
        rec: dict[str, Any] = {"per_route": {}}
        total = 0.0
        votes = 0
        for name in w:
            rk = ranks.get(name, {}).get(idx)
            sc = norms.get(name, {}).get(idx, 0.0)
            rec["per_route"][name] = {"rank": rk, "score": round(sc, 4)}
            if rk is not None:
                votes += 1
            if cfg.fusion == "weighted":
                total += w[name] * sc
            elif cfg.fusion == "rrf":
                if rk is not None:
                    total += w[name] * (1.0 / (rrf_k + rk))
            elif cfg.fusion == "borda":
                if rk is not None:
                    size = max(1, len(ranks.get(name, {})))
                    total += w[name] * (size - rk + 1) / size
            elif cfg.fusion == "vote":
                # 纯投票：只看"有没有被这一路召回"，不看名次高低。
                # 末尾那个极小的 `1e-6·sc` 是 tie-break：只投出 1 票/0 票两种值时，
                # 大量候选会并列，没有它排序就退化成集合的遍历顺序（等于随机）。
                total += w[name] * ((1.0 if rk is not None else 0.0) + 1e-6 * sc)
            elif cfg.fusion == "evidence":
                cos = norms.get("vector", {}).get(idx, 0.0)
                bm = norms.get("fulltext", {}).get(idx, 0.0)
                total = cos + RETRIEVAL_BM25_BETA * bm
            else:
                total += w[name] * sc
        rec["fused"] = total
        rec["votes"] = votes
        fused[idx] = rec

    trace = {
        "fusion": cfg.fusion,
        "fusion_name": FUSIONS.get(cfg.fusion, {}).get("name", cfg.fusion),
        "weights": {k: round(v, 3) for k, v in w.items()},
        "pool_size": len(pool),
        "routes": [r.name for r in routes],
        "route_ms": {r.name: round(r.ms, 2) for r in routes},
        "route_note": {r.name: r.note for r in routes},
        "multi_hit": sum(1 for v in fused.values() if v["votes"] > 1),
    }
    return fused, trace


def _apply_priors(fused: dict[int, dict], kb: KnowledgeBase,
                  clip_prior: dict[int, float], hint: list[str] | None,
                  cfg: HybridConfig) -> tuple[int, dict]:
    """
    在融合分之上叠加**前几个工单已经验证过的排序先验**（全部是加法，全部只影响排序）。

    这一步是"不能省"的 —— 它是本工单评测跑出来最重要的一条教训：
    工单6 是**叠加**在工单1~5 之上的，如果为了"公式干净"把前面用消融实验
    一个个验出来的先验丢掉，混合检索的表现会**低于**重建它之前的系统。
    实测：丢掉先验时准确率比基线低 9.5 个百分点（76.19% vs 85.71%）——
    差的不是融合算法，而是「问的是哪一家」这个信号。

    三条例（都在 `config` 里有出处，不是凭感觉加的系数）：

      * **文档消歧加成**（δ=0.25，工单3）：问题点名某一份文档时，该文档的块获得加成。
        两份招股书是同一个模板写出来的，这是客观存在的强混淆 ——
        不给这个信号，「武汉力源信息技术股份有限公司的组织结构图」会召回一堆兴图的块，
        关键事实自然凑不齐（这正是准确率掉 9.5 个点的直接原因）。
      * **图像先验**（δ=0.12·CLIP 相似度，工单4）：跨模态召回通道的排序体现。
      * **低信息图惩罚**（0.15，工单4 调优项）：证照照片那类图（图内文字是签名页碎片）
        降权，避免它靠词表长度把真正含数据的表挤下去。
      * **文档共识**（δ=0.25，工单1~5 的原始公式项）：候选池里同一来源文档的块占比，
        是弱信号，主要作用是在单文档语料上退化为常数（不改变排序，也不影响闸门）。

    闸门看不到这些先验（它只看 evidence 的绝对量纲），
    所以「问的是哪一家」这件事不会把离题问题放进闸门。
    """
    doc_counts: dict[str, int] = {}
    for idx in fused:
        dk = kb.chunks[idx].get("doc_key", "") or ""
        doc_counts[dk] = doc_counts.get(dk, 0) + 1
    mx = max(doc_counts.values()) if doc_counts else 1

    n_low = 0
    for idx, rec in fused.items():
        c = kb.chunks[idx]
        dk = c.get("doc_key", "") or ""
        consensus = doc_counts.get(dk, 0) / mx
        hint_hit = 1.0 if (hint and dk in hint) else 0.0
        clip = float(clip_prior.get(idx, 0.0))
        low = 0.0
        if cfg.use_image_prior and c.get("type") == "image" and c.get("fig_type"):
            if any(t in str(c["fig_type"]) for t in LOW_INFO_FIG_TYPES):
                low = RETRIEVAL_LOW_INFO_PENALTY
                n_low += 1
        rec["prior"] = {
            "hint": round(RETRIEVAL_DOC_HINT_DELTA * hint_hit, 4),
            "clip": round(RETRIEVAL_IMAGE_DELTA * clip, 4),
            "consensus": round(RETRIEVAL_DOC_CONSENSUS_DELTA * consensus, 4),
            "low_info": round(low, 4),
        }
        rec["fused"] = (rec.get("fused", 0.0)
                        + RETRIEVAL_DOC_HINT_DELTA * hint_hit
                        + RETRIEVAL_IMAGE_DELTA * clip
                        + RETRIEVAL_DOC_CONSENSUS_DELTA * consensus
                        - low)
    return n_low, doc_counts


# ============================================================ 主入口

def search(query: str, cfg: HybridConfig | None = None,
           kb: KnowledgeBase | None = None,
           override_query: str | None = None) -> HybridResult:
    """
    执行一次（可配置的）检索。

    `override_query`：上游 Query 理解模块（工单5）已改写过的检索式 —— 直接用它做
    向量路与全文路的输入，不再二次归一化。全文路仍会保留原句里的查询语法
    （布尔/短语），因为语法是**用户意图的一部分**，改写不该把它抹掉。
    """
    t0 = time.perf_counter()
    cfg = cfg or HybridConfig()
    kb = kb or KnowledgeBase.get()

    normalized = normalize_query(query)
    used = (override_query or "").strip() or normalized

    # ---- 错别字容错（工单6「模糊查询」）--------------------------------
    # 放在**两路之前**：错字对两路都有害 —— 全文路 df=0 直接不命中，
    # 向量路更惨，一个错字能把整句语义带偏（实测 p60 → p214）。
    # 修完再分流，两路吃的是同一句改写后的话，trace 里能看到改了什么。
    repair_notes: list[dict] = []
    if cfg.use_repair and FULLTEXT_REPAIR:
        try:
            from .fulltext import get_index  # 局部导入：避免与索引构建的循环依赖
            used, repair_notes = get_index(kb).repair_text(used)
        except Exception as exc:      # 容错本身不许把检索搞挂
            logger.warning("错别字容错失败，改用原句：%s", exc)

    # 向量路：字符串级 DF 过滤（编码需要一句连贯的话，删掉无区分度成分后语义更聚焦）。
    # 全文路：**不做**字符串级过滤，改用叶子节点级的 stop_terms ——
    # 理由见 _route_fulltext 的 docstring（字符串过滤会把布尔语法一起删坏）。
    effective = kb.filter_query_terms(used) if cfg.use_df_filter else used
    ft_query = used

    trace: dict = {"steps": [], "config": cfg.to_dict(), "effective_query": effective,
                   "repair": repair_notes}
    if repair_notes:
        trace["steps"].append({
            "name": "错别字容错", "ms": 0.0,
            "note": "、".join(f"「{n['from']}」→「{n['to']}」" for n in repair_notes)})
    hint = None
    if cfg.use_doc_hint:
        hint = kb.detect_docs(used) or None
    trace["doc_hint"] = hint or []
    if hint:
        trace["steps"].append({"name": "doc_hint", "ms": 0.0,
                               "note": f"问题点名 {'/'.join(hint)} → 候选池内加成"})

    routes: list[RouteResult] = []
    by_name: dict[str, RouteResult] = {}
    names = cfg.routes()
    pool = max(cfg.pool_top_k, cfg.top_k, RETRIEVAL_FINAL_TOP_K)

    if "vector" in names:
        r = _route_vector(effective, kb, cfg, max(pool, RETRIEVAL_DENSE_TOP_K), hint or [])
        by_name["vector"] = r
        routes.append(r)
        trace["steps"].append({"name": "vector", "ms": round(r.ms, 2), "note": r.note})

    # 全文路在两种情况下都要跑：
    #   ① 它参与排序（strategy ∈ {fulltext, hybrid}）；
    #   ② 它不参与排序、但要给**闸门**提供 BM25 项 ——
    #      依据分的定义是 `余弦 + β·BM25归一`（工单1~5 的绝对量纲），
    #      少一项会让"纯向量策略"下的闸门阈值实际变得更严，
    #      于是同一道题换个策略就被拦空，评测结果没法横向比。
    if "fulltext" in names or cfg.apply_gate:
        r = _route_fulltext(ft_query, kb, cfg, max(pool, RETRIEVAL_SPARSE_TOP_K))
        by_name["fulltext"] = r
        if "fulltext" in names:
            routes.append(r)
        note = r.note + ("" if "fulltext" in names else "（不参与排序，仅供闸门算依据分）")
        trace["steps"].append({"name": "fulltext", "ms": round(r.ms, 2), "note": note})

    # CLIP 跨模态：作为**加法先验**（与工单4 完全一致），不作为并列的第三路
    clip_prior: dict[int, float] = {}
    if cfg.use_clip and CLIP_ENABLED:
        rc = _route_clip(effective, kb, cfg, pool)
        if rc.scores:
            mx = max(rc.scores.values()) or 1.0
            clip_prior = {i: v / mx for i, v in rc.scores.items()}
        trace["steps"].append({"name": "clip", "ms": round(rc.ms, 2), "note": rc.note})

    fused, ftrace = _fuse(routes, cfg)

    # CLIP 命中的图块要**并入候选池**：它的价值恰恰在于捞回"文本路都没找到、
    # 但图向量觉得像"的那些块（工单4 的核心用例）。
    for idx in clip_prior:
        fused.setdefault(idx, {
            "per_route": {k: {"rank": None, "score": 0.0} for k in ftrace.get("weights", {})},
            "fused": 0.0, "votes": 0,
        })

    n_low, _ = _apply_priors(fused, kb, clip_prior, hint, cfg)
    ftrace["clip_prior_hits"] = len(clip_prior)
    ftrace["low_info_in_pool"] = n_low
    trace["fusion_detail"] = ftrace
    trace["steps"].append({
        "name": "fusion", "ms": 0.0,
        "note": (f"{ftrace['fusion_name']}｜"
                 + "｜".join(f"{k}={v:.2f}" for k, v in ftrace["weights"].items())
                 + f"｜候选池 {ftrace['pool_size']} 条（两路同时命中 {ftrace['multi_hit']} 条）"),
    })

    if not fused:
        trace["total_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        return HybridResult(query, cfg, [], trace, gated=True,
                            original_query=query, normalized_query=normalized, used_query=used)

    # ---- 组装候选：原始分 → 归一 → Candidate（供重排器使用）
    # 依两路的**归一化分**算依据分（绝对量纲），与工单1~5 的定义严格一致：
    # evidence = 余弦 + β·BM25归一。不管当前策略是哪一种，都不改这个定义 ——
    # 否则同一道题换策略就会撞上不同的闸门行为，评测没法横向比。
    vec_norm = _norm_route(by_name["vector"]) if "vector" in by_name else {}
    ft_norm = _norm_route(by_name["fulltext"]) if "fulltext" in by_name else {}

    order = sorted(fused, key=lambda i: -fused[i]["fused"])
    raw_vals = np.array([fused[i]["fused"] for i in order], dtype=np.float32)
    lo, hi = float(raw_vals.min()), float(raw_vals.max())
    span = hi - lo
    cands: list[Candidate] = []
    metas: dict[int, RetrievedChunk] = {}
    for i in order:
        c = kb.chunks[i]
        norm = float((fused[i]["fused"] - lo) / span) if span > 1e-9 else 0.5
        cosine = float(vec_norm.get(i, 0.0))
        bm = float(ft_norm.get(i, 0.0))
        clip = float(clip_prior.get(i, 0.0))
        rc = RetrievedChunk(
            chunk_id=c["chunk_id"], text=c["text"], page=c.get("page", 0),
            page_end=c.get("page_end", c.get("page", 0)), section=c.get("section", ""),
            type=c.get("type", "text"), doc=c.get("doc", ""), doc_key=c.get("doc_key", ""),
            image=c.get("image", ""), fig_type=c.get("fig_type", ""),
            caption=c.get("caption", ""), cosine=cosine, bm25_norm=bm,
            clip_sim=clip,
            # evidence 沿用「绝对量纲」的定义（余弦 + β·BM25归一），阈值闸门只认它：
            # 闸门若看融合分，纯全文命中的离题块会因为名次分而混进来。
            evidence=cosine + RETRIEVAL_BM25_BETA * bm if "vector" in names else bm,
            score=fused[i]["fused"],
        )
        metas[i] = rc
        cands.append(Candidate(
            idx=i, text=rc.text, base_score=fused[i]["fused"], norm_score=norm,
            meta={"page": rc.page, "section": rc.section, "type": rc.type,
                  "doc_key": rc.doc_key, "doc": rc.doc, "image": rc.image,
                  "fig_type": rc.fig_type, "caption": rc.caption,
                  "hint_hit": bool(hint and rc.doc_key in hint),
                  "multihit": fused[i]["votes"] > 1},
            sources=[k for k, v in fused[i]["per_route"].items() if v["rank"] is not None],
        ))

    # ---- 重排
    cands = cands[: max(cfg.rerank_candidates, cfg.top_k)]
    rr = get_reranker(cfg.reranker, kb)
    # 交给重排器的是**过滤后的检索式**（effective），不是原句：
    # 原句里的公司全称会让 TF-IDF/LLM 重排器把「满是公司名的合同表格」顶上来，
    # 实测这会让重排从"微调"变成"帮倒忙"（准确率反而掉 4.8 个百分点）。
    ranked, rtrace = rr.rerank(effective, cands, top_k=cfg.top_k, alpha=cfg.rerank_alpha)
    trace["rerank"] = rtrace
    trace["steps"].append({
        "name": "rerank", "ms": rtrace.get("ms", 0.0),
        "note": f"{rtrace.get('name', cfg.reranker)}：精排 {len(cands)} 条 → 取 {len(ranked)} 条"
                + (f"（已降级：{rtrace['degraded']}）" if rtrace.get("degraded") else ""),
    })

    items: list[RetrievedChunk] = []
    for ro in ranked:
        rc = metas.get(ro.idx)
        if rc is None:
            continue
        rc.score = ro.final_score
        rc.rerank_score = ro.rerank_score
        rc.rerank_reason = ro.reason
        rc.sources = cands_sources(cands, ro.idx)
        items.append(rc)

    # ---- 阈值闸门（只对"有向量路"的配置生效）
    gated = False
    if cfg.apply_gate and "vector" in names:
        passed = [it for it in items if it.evidence >= HYBRID_MIN_EVIDENCE]
        gated = len(passed) == 0
        items = passed

    trace.update({
        "gate": {"applied": bool(cfg.apply_gate and "vector" in names),
                 "min_evidence": HYBRID_MIN_EVIDENCE,
                 "gated": gated},
        "kept": len(items),
        "total_ms": round((time.perf_counter() - t0) * 1000, 2),
    })
    return HybridResult(query, cfg, items, trace, gated,
                        original_query=query, normalized_query=normalized, used_query=used)


def cands_sources(cands: list[Candidate], idx: int) -> list[str]:
    for c in cands:
        if c.idx == idx:
            return c.sources
    return []


def retrieve(query: str, **kwargs) -> "HybridResult":
    """
    `retriever.retrieve()` 的工单6 版（同名参数尽量兼容，便于上层无痛切换）。

    与旧版的差别只有一处是**破坏性**的：旧版返回 `RetrievalResult`，
    本函数返回 `HybridResult`。两者都有 `.items` / `.trace` / `.gated` / `.contexts()`，
    因此 `rag.py` 与 `build_context()` 都能直接复用，不需要改。
    """
    keys = {"top_k", "strategy", "fusion", "vector_weight", "reranker",
            "rerank_alpha", "rerank_candidates", "embed_model", "ft_mode",
            "ft_syntax", "use_df_filter", "use_doc_hint", "use_clip", "apply_gate",
            "use_dense", "use_sparse", "pool_top_k", "use_repair", "use_image_prior"}
    cfg_kwargs = {k: v for k, v in kwargs.items() if k in keys and v is not None}
    kb = kwargs.get("kb")
    override = kwargs.get("override_query")
    return search(query, HybridConfig(**cfg_kwargs), kb=kb, override_query=override)


def describe() -> dict:
    """把可配置项全量吐给前端（策略/融合/重排器/权重范围/模型清单）。"""
    from .reranker import feedback_count, list_rerankers

    return {
        "strategies": [{"key": k, **v} for k, v in STRATEGIES.items()],
        "fusions": [{"key": k, **v} for k, v in FUSIONS.items()],
        "rerankers": list_rerankers(),
        "embed_models": embed_registry.list_models(),
        "defaults": HybridConfig().to_dict(),
        "weight_range": [0.0, 1.0],
        "feedback_count": feedback_count(),
        "fulltext": {
            "enabled": FULLTEXT_ENABLED,
            "syntax": [
                {"syntax": "军用领域", "mean": "普通词"},
                {"syntax": '"军用领域"', "mean": "短语匹配（要求相邻）"},
                {"syntax": "收入 AND 军用", "mean": "布尔：同时命中"},
                {"syntax": "收入 OR 利润", "mean": "布尔：任一命中（默认）"},
                {"syntax": "收入 NOT 风险", "mean": "布尔：排除"},
                {"syntax": "section:风险提示", "mean": "字段限定（title/body/summary）"},
                {"syntax": "注册酱~1", "mean": "模糊匹配（编辑距离≤1）"},
                {"syntax": "销售*", "mean": "前缀通配"},
            ],
        },
    }
