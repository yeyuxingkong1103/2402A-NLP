"""
检索模块（稠密 + 稀疏 加权融合）
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
工单编号：人工智能NLP-RAG-图像内容解析及检索优化

对应工单功能需求「检索与生成 → 向量检索：将文档内容嵌入向量空间，支持高效检索」。

融合公式（**刻意不用裸 RRF**，理由见下）：

    依据分 evidence = 余弦 + β · BM25归一          β = 0.30
    融合分 score    = 依据分 + δ · 文档共识         δ = 0.25
    闸门            依据分 ≥ min_evidence           默认 0.55

三个设计要点：

1) **BM25 项必须先归一化**。BM25 无上界（本语料实测 0~20+），余弦在 [0,1]，
   直接相加会被 BM25 淹没，且结果随语料规模漂移。用组内最大值归一到 0~1。

2) **主题先验用加法，不用乘法**。乘法对「余弦高但没命中关键词」的块是灾难
   （0.72 × 0.5 = 0.36，反而输给余弦 0.68 但有关键词加成的块）。
   加法保留余弦召回的价值。

3) **双分是刻意的**：依据分有绝对量纲，阈值闸门只认它；融合分含主题先验，
   只用于排序与展示。若让闸门看融合分，离题问题会因为先验加分而拦不住。

另外不用 RRF 的原因：RRF 只看排名、无视分数，会把「仅被 BM25 命中的纯关键词噪音」
按名次奖励上来；且归一化 RRF 的 top1 恒为 1.0，阈值闸门形同虚设。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .config import (
    CLIP_ENABLED,
    LOW_INFO_FIG_TYPES,
    RETRIEVAL_LOW_INFO_PENALTY,
    RETRIEVAL_BM25_BETA,
    RETRIEVAL_BM25_REF_PERCENTILE,
    RETRIEVAL_CLIP_MIN_SIM,
    RETRIEVAL_CLIP_TOP_K,
    RETRIEVAL_DENSE_TOP_K,
    RETRIEVAL_DOC_CONSENSUS_DELTA,
    RETRIEVAL_DOC_HINT_DELTA,
    RETRIEVAL_FINAL_TOP_K,
    RETRIEVAL_IMAGE_DELTA,
    RETRIEVAL_MIN_EVIDENCE,
    RETRIEVAL_SPARSE_TOP_K,
    WORK_ORDER_NO,  # noqa: F401
)
from .embedder import cosine_scores, embed_query
from .index_store import KnowledgeBase
from .query_norm import normalize_query


@dataclass
class RetrievedChunk:
    """一条召回结果。"""

    chunk_id: str
    text: str
    page: int
    page_end: int
    section: str
    type: str
    doc: str
    doc_key: str = ""        # 来源文档短键（xingtu / liyuan）—— 两文档语料下用于消歧
    # ---- 工单04：图像块透传字段（非图像块为空），前端据此回显裁剪图 ----
    image: str = ""          # 裁剪图相对路径
    fig_type: str = ""       # 图类型
    caption: str = ""        # 图题
    cosine: float = 0.0
    bm25_raw: float = 0.0
    bm25_norm: float = 0.0
    evidence: float = 0.0   # 依据分：余弦 + β·BM25归一（不含主题先验）
    consensus: float = 0.0  # 文档共识先验 0~1
    clip_sim: float = 0.0   # CLIP 图文相似度（0 = 该块没走 CLIP 通道）
    score: float = 0.0      # 融合分：依据分 + δ·共识（**可以 > 1，不是相似度**）
    # ---- 工单06：混合检索新增的透传字段 ----
    #   sources        该块被哪几路召回（vector / fulltext / clip）——
    #                  演示时"这条是向量路找到的、那条是全文路找到的"靠它显示
    #   rerank_score   重排器给出的原始分（0~1）
    #   rerank_reason  重排器给出的理由（TF-IDF 是特征分解，LLM 是模型给的一句话）
    sources: list[str] = field(default_factory=list)
    rerank_score: float = 0.0
    rerank_reason: str = ""

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "page": self.page,
            "page_end": self.page_end,
            "section": self.section,
            "type": self.type,
            "doc": self.doc,
            "doc_key": self.doc_key,
            "cosine": round(self.cosine, 4),
            "bm25_raw": round(self.bm25_raw, 4),
            "bm25_norm": round(self.bm25_norm, 4),
            "evidence": round(self.evidence, 4),
            "consensus": round(self.consensus, 4),
            "clip_sim": round(self.clip_sim, 4),
            "score": round(self.score, 4),
            "image": self.image,
            "fig_type": self.fig_type,
            "caption": self.caption,
            "sources": self.sources,
            "rerank_score": round(self.rerank_score, 4),
            "rerank_reason": self.rerank_reason,
        }


@dataclass
class RetrievalResult:
    """一次检索的完整结果 + 链路 trace（前端「检索链路」面板用）。"""

    original_query: str
    normalized_query: str
    used_query: str
    items: list[RetrievedChunk] = field(default_factory=list)
    gated: bool = False          # True 表示被阈值拦空
    trace: dict = field(default_factory=dict)

    def contexts(self) -> list[str]:
        return [it.text for it in self.items]

    def to_dict(self) -> dict:
        return {
            "original_query": self.original_query,
            "normalized_query": self.normalized_query,
            "used_query": self.used_query,
            "gated": self.gated,
            "trace": self.trace,
            "items": [it.to_dict() for it in self.items],
        }


def _fig_map_from_kb(kb: "KnowledgeBase") -> tuple[dict[tuple, int], dict[int, int]]:
    """建立「图 → 块」的两张映射表。

    返回 (按图身份, 按 CLIP 行号)：
      * 按图身份 `(doc_key, fig_page, fig_index)` —— **主用**。
        这张表的键在图清单与分块里都能直接取到，因此「算完 CLIP 向量」不必回头
        重建索引：向量按行对应图清单，图清单再按身份对应到块。
      * 按 `clip_index` —— 备用。索引里若已写死行号（重建过），命中更快，
        但它会随「重算向量」而失效，所以只当兜底。

    一张图的描述可能被切成多块（"结构关系"太长时按节分块），但 CLIP 向量只有一条。
    命中时只把**首块**拉进候选池：首块带完整定位头（图类型 + 图题），
    把同图的所有分块一起加进来会挤占 top-k 名额，且它们是同一份信息。
    """
    by_ident: dict[tuple, int] = {}
    by_row: dict[int, int] = {}
    for i, c in enumerate(kb.chunks):
        if c.get("type") != "image":
            continue
        ident = (c.get("doc_key", ""), c.get("fig_page", c.get("page", 0)),
                 c.get("fig_index", -1))
        by_ident.setdefault(ident, i)
        ci = c.get("clip_index", -1)
        if isinstance(ci, int) and ci >= 0:
            by_row.setdefault(ci, i)
    return by_ident, by_row


_CLIP_CACHE: dict = {"key": None, "vecs": None, "figures": []}


def _clip_assets():
    """读 CLIP 矩阵 + 图清单（带 mtime 缓存）。

    每次检索都 np.load 一份几十 KB~几 MB 的矩阵是纯浪费，因此按文件 mtime 缓存；
    重新构建索引（重算向量）后 mtime 变化会自动失效，不会读到错位的旧矩阵。
    """
    if not CLIP_ENABLED:
        return None, []
    from .config import CLIP_EMB_NPY, FIGURES_JSON

    try:
        key = (CLIP_EMB_NPY.stat().st_mtime_ns, FIGURES_JSON.stat().st_mtime_ns)
    except OSError:
        return None, []
    if _CLIP_CACHE["key"] == key:
        return _CLIP_CACHE["vecs"], _CLIP_CACHE["figures"]
    from .clip_encoder import load_clip_index

    vecs, figures = load_clip_index()
    _CLIP_CACHE.update({"key": key, "vecs": vecs, "figures": figures})
    return vecs, figures


def retrieve(
    query: str,
    top_k: int | None = None,
    override_query: str | None = None,
    apply_gate: bool = True,
    use_sparse: bool = True,
    use_df_filter: bool = True,
    kb: "KnowledgeBase | None" = None,
    doc_hint: list[str] | None = None,
    use_doc_hint: bool = True,
    use_clip: bool = True,
    use_image_prior: bool = True,
) -> RetrievalResult:
    """
    执行一次混合检索。

    override_query：调用方（Query 理解模块）已给出改写后的检索式时直接用它，
    避免二次归一化把改写结果再削一遍。

    use_sparse / use_df_filter / apply_gate：**工单02 消融实验开关**。
    三个开关默认全开 = 当前主线系统；逐个关闭即可回退到检索层的早期形态，
    用于量化「BM25 混合」「DF 过滤」「阈值闸门」各自的贡献（见 scripts/ablation_retrieval.py）。

    kb：显式指定知识库，默认取全局单例。消融实验用它注入「同一套检索算法跑在
    另一份分块上」的对照索引 —— 这样切换分块方式时，检索逻辑**一行都不用复制**，
    也就不会出现「对照组用的是另一套融合公式」这种不自洽。

    doc_hint / use_doc_hint：**工单03 多文档消歧**。
    DF 过滤会剔掉公司全称，而两文档语料里公司名恰恰是唯一能区分「答的是哪一家」的信号。
    做法是在**原问题**里识别公司名 → 把候选池限定到那份文档（检索式本身照旧过滤）。
    显式传 doc_hint 时不猜；传 None 且 use_doc_hint=True 时自动识别；
    识别不到、或限定后候选为空 → 不做限定（宁可放宽，也不能把答案锁死在外面）。

    use_clip：**工单04 CLIP 跨模态召回通道**（默认开，CLIP 不可用时自动降级）。
    它解决的是纯文本检索的一个固有短板：问题的**措辞**与图的**文本描述**对不上。
    例如问「哪个行业的增长率是负数」，而描述里写的是「IC 卡：−2.0%」——
    「负数」和「−2.0%」在字面与向量空间上都不近，纯文本检索会漏；
    但 CLIP 把「增长率是负数的行业」这句话直接编码成向量，与那张柱状图的图向量
    在同一空间里比对，仍可能把它排上来。
    设计上仍是**加法先验、只作用于 score**：闸门只看 evidence（绝对量纲），
    否则「长得像图」这件事会把离题问题放进闸门。CLIP 相似度还会按组内最大值
    归一到 0~1 再乘权重 —— 与 BM25 项同一套处理，避免随语料规模漂移。

    use_image_prior：**工单04 调优开关**。对「低信息量图」（证照照片这类，
    图内文字是签名/承诺页碎片）给一个小幅排序惩罚。开与关的差别由消融实验的
    T1/T2 两条臂量化（见 scripts/ablation_image.py）——这不是凭感觉加的系数，
    是为了修掉「一张证照照片把 id=531 的正确答案挤到第 2 名」这一处实测代价。
    """
    t0 = time.perf_counter()
    kb = kb or KnowledgeBase.get()
    final_k = top_k or RETRIEVAL_FINAL_TOP_K
    n = len(kb.chunks)

    normalized = normalize_query(query)
    used = (override_query or "").strip() or normalized
    # 再剔一遍「语料级无区分度词」（公司全称等）。两路召回都用过滤后的式子，
    # 否则公司全称会把向量和 BM25 同时往「满是公司名的合同表格」上带偏。
    effective = kb.filter_query_terms(used) if use_df_filter else used

    trace: dict = {"steps": []}
    if use_df_filter and effective != used:
        trace["steps"].append(
            {"name": "filter", "ms": 0.0, "note": f"剔除无区分度词 → {effective}"}
        )

    # ---- 0) 多文档消歧：用**原问题**（未被 DF 过滤掉公司名的那一份）判断问的是哪一家
    hint = doc_hint
    if hint is None and use_doc_hint:
        hint = kb.detect_docs(query)
    hint = [h for h in (hint or []) if h in kb.doc_keys] if kb.doc_keys else []
    trace["doc_hint"] = hint
    if hint:
        trace["steps"].append(
            {"name": "doc_hint", "ms": 0.0,
             "note": f"问题点名了 {len(hint)} 份文档 → 候选池限定为 {'/'.join(hint)}"}
        )

    if n == 0:
        return RetrievalResult(query, normalized, used, [], True, trace)

    # ---- 1) 稠密召回
    t = time.perf_counter()
    qvec = embed_query(effective)
    dense = cosine_scores(kb.embeddings, qvec)
    dense_ms = (time.perf_counter() - t) * 1000
    trace["steps"].append({"name": "dense", "ms": round(dense_ms, 2), "note": f"余弦全量 {len(dense)} 条"})

    # ---- 2) 稀疏召回（可关）
    t = time.perf_counter()
    if use_sparse:
        sparse = kb.bm25_scores(effective)
        sparse_ms = (time.perf_counter() - t) * 1000
        bm25_hits = int((sparse > 0).sum())
        trace["steps"].append(
            {"name": "sparse", "ms": round(sparse_ms, 2), "note": f"BM25 命中 {bm25_hits} 条"}
        )
    else:
        sparse = np.zeros(n, dtype=np.float32)
        trace["steps"].append({"name": "sparse", "ms": 0.0, "note": "已关闭（消融实验）"})

    # ---- 3) 候选池：两路各取 top-k 后求并集
    dense_idx = np.argsort(-dense)[: max(RETRIEVAL_DENSE_TOP_K, final_k)]
    if use_sparse:
        sparse_idx = np.argsort(-sparse)[: max(RETRIEVAL_SPARSE_TOP_K, final_k)]
        pool = sorted(set(dense_idx.tolist()) | set(sparse_idx.tolist()))
    else:
        # 关掉 BM25 时**不能**沿用 argsort(全零) —— 那会返回下标 0..k-1，
        # 把最前面几块无脑塞进候选池，凭空引入一段固定噪音。
        pool = sorted(set(dense_idx.tolist()))

    # ---- 3b) 文档消歧：问题点名了某一份文档 → ① 保证那一份的最优块进候选池
    #          ② 给它一个排序加成（加法先验，与「文档共识」同一套设计）
    #
    # 只做「把候选池限定在该文档内」是不够的（试过，会翻车）：两路召回的 top-k 里
    # 可能根本没有那份文档的块，限定之后只剩个位数候选，反而把正确答案锁在门外。
    # 正确做法是**在候选池之外，再按文档内排名补一轮召回**，再对该文档的块加权。
    # 加成只作用于 score（排序用），闸门仍然只看 evidence（绝对量纲），
    # 所以「问的是哪一家」不会把离题问题放进闸门。
    hint_bonus = 0.0
    if hint:
        mask = np.array([c.get("doc_key") in hint for c in kb.chunks], dtype=bool)
        before = len(pool)
        add: set[int] = set()
        for scores in ([dense] + ([sparse] if use_sparse else [])):
            scoped = np.where(mask, scores, -np.inf)
            add |= set(np.argsort(-scoped)[: max(RETRIEVAL_DENSE_TOP_K, final_k)].tolist())
        add = {i for i in add if mask[i]}
        pool = sorted(set(pool) | add)
        hint_bonus = RETRIEVAL_DOC_HINT_DELTA
        trace["steps"].append(
            {"name": "doc_hint", "ms": 0.0,
             "note": f"命中 {'/'.join(hint)} → 候选池 {before} → {len(pool)} 条（含文档内召回）"}
        )

    # ---- 3c) CLIP 跨模态召回：以文搜图（工单04）
    #
    # 与 3b 同构，但召回的**不是**「问题像不像这段文字」，而是
    # 「问题像不像这张图」。命中后做两件事：① 把该图的首块拉进候选池
    # （给它一个进池的机会，哪怕两路文本召回都没捞到它）；
    # ② 记一个加法先验，在排序时体现跨模态证据。
    clip_hits: dict[int, float] = {}     # chunk 下标 → 归一化相似度
    clip_note = "已关闭（消融实验）"
    if use_clip:
        vecs, figures = _clip_assets()
        if vecs is None or not figures:
            clip_note = "跳过（无 CLIP 向量，检索链路照常）"
        else:
            t = time.perf_counter()
            try:
                from .clip_encoder import ClipEncoder

                qv = ClipEncoder.instance().encode_texts([effective])
                sims = (vecs @ qv[0]).astype(np.float32)      # 两边都已 L2 归一 → 内积即余弦
                order = np.argsort(-sims)[:RETRIEVAL_CLIP_TOP_K]
                top_sim = float(sims[order[0]]) if len(order) else 0.0
                # 地板值：CLIP 相似度恒为正，不加地板等于给全库图无条件发先验
                keep = [int(r) for r in order if float(sims[r]) >= RETRIEVAL_CLIP_MIN_SIM]
                by_ident, by_row = _fig_map_from_kb(kb)
                for r in keep:
                    fig = figures[r] if r < len(figures) else {}
                    ci = by_ident.get((fig.get("doc_key", ""), fig.get("page", 0),
                                       fig.get("index", -1)))
                    if ci is None:
                        ci = by_row.get(r)      # 兜底：索引里写死了行号时走这条
                    if ci is None:
                        continue
                    norm = (float(sims[r]) / top_sim) if top_sim > 0 else 0.0
                    clip_hits[ci] = max(clip_hits.get(ci, 0.0), norm)
                pool = sorted(set(pool) | set(clip_hits))
                clip_note = (f"命中 {len(keep)} 张图（相似度≥{RETRIEVAL_CLIP_MIN_SIM}，"
                             f"top={top_sim:.3f}，并入候选池 {len(clip_hits)} 块）")
                clip_ms = (time.perf_counter() - t) * 1000
            except Exception as exc:  # noqa: BLE001 —— 可选通道，失败不影响主链路
                clip_ms = (time.perf_counter() - t) * 1000
                clip_note = f"跳过（{type(exc).__name__}: {str(exc)[:60]}）"
            trace["steps"].append({"name": "clip", "ms": round(clip_ms, 2), "note": clip_note})
    else:
        trace["steps"].append({"name": "clip", "ms": 0.0, "note": clip_note})

    # ---- 4) 加权融合
    # BM25 归一化的基准见 config.RETRIEVAL_BM25_NORM 的说明：
    # 默认按**候选池内最大值**归一，让 evidence 的量纲与全库规模解耦。
    if use_sparse and sparse.size:
        # 参考基准用**分位数**（默认 p95）而不是最大值：max 会被单个离群块带走，
        # 详见 config.RETRIEVAL_BM25_REF_PERCENTILE 的说明。
        ref_pool = sparse[pool] if pool else sparse
        max_bm25 = (float(np.percentile(ref_pool, RETRIEVAL_BM25_REF_PERCENTILE))
                    if RETRIEVAL_BM25_REF_PERCENTILE < 100.0 else float(sparse.max()))
    else:
        max_bm25 = 0.0
    beta = RETRIEVAL_BM25_BETA if use_sparse else 0.0
    items: list[RetrievedChunk] = []
    for i in pool:
        c = kb.chunks[i]
        cos = float(dense[i])
        b_raw = float(sparse[i])
        # 截断到 1.0：分位点以上的块视为"关键词相关度已拉满"，维持原有的 0~1 契约
        b_norm = min(1.0, b_raw / max_bm25) if max_bm25 > 0 else 0.0
        ev = cos + beta * b_norm
        items.append(
            RetrievedChunk(
                chunk_id=c["chunk_id"],
                text=c["text"],
                page=c.get("page", 0),
                page_end=c.get("page_end", c.get("page", 0)),
                section=c.get("section", ""),
                type=c.get("type", "text"),
                doc=c.get("doc", ""),
                doc_key=c.get("doc_key", ""),
                image=c.get("image", ""),
                fig_type=c.get("fig_type", ""),
                caption=c.get("caption", ""),
                cosine=cos,
                bm25_raw=b_raw,
                bm25_norm=b_norm,
                evidence=ev,
                clip_sim=clip_hits.get(i, 0.0),
            )
        )

    # ---- 5) 文档级共识先验 + 文档消歧加成（都是加法，都只影响排序）
    # 统计候选池里每个来源文档命中的块数；单文档语料下所有块共识相同，
    # 相当于加一个常数，不改变排序，也不会误伤闸门（闸门只看依据分）。
    doc_counts: dict[str, int] = {}
    for it in items:
        doc_counts[it.doc] = doc_counts.get(it.doc, 0) + 1
    max_doc_hit = max(doc_counts.values()) if doc_counts else 1
    n_low_info = 0
    for it in items:
        it.consensus = doc_counts.get(it.doc, 0) / max_doc_hit
        bonus = hint_bonus if (hint and it.doc_key in hint) else 0.0
        clip_bonus = RETRIEVAL_IMAGE_DELTA * it.clip_sim
        # 低信息量图（证照照片…）的排序惩罚：只减 score，不动 evidence。
        # 判据用 `in`（子串包含）而不是相等 —— 图类型是多模态模型自由生成的文本，
        # 「证照照片」「营业执照照片」这类变体都该归到同一类里。
        penalty = 0.0
        if use_image_prior and it.type == "image" and it.fig_type:
            if any(t in it.fig_type for t in LOW_INFO_FIG_TYPES):
                penalty = RETRIEVAL_LOW_INFO_PENALTY
                n_low_info += 1
        it.score = (it.evidence + RETRIEVAL_DOC_CONSENSUS_DELTA * it.consensus
                    + bonus + clip_bonus - penalty)

    # ---- 6) 阈值闸门（只看依据分，可关）
    items.sort(key=lambda x: -x.score)
    if apply_gate:
        passed = [it for it in items if it.evidence >= RETRIEVAL_MIN_EVIDENCE]
        gated = len(passed) == 0
        items = passed
    else:
        gated = False

    items = items[:final_k]

    evs = [it.evidence for it in items] or [0.0]
    trace["fusion"] = {
        "beta": beta,
        "delta": RETRIEVAL_DOC_CONSENSUS_DELTA,
        "doc_hint": hint,
        "doc_hint_delta": hint_bonus,
        "image_delta": RETRIEVAL_IMAGE_DELTA,
        "clip_hits": len(clip_hits),
        "clip_top_k": RETRIEVAL_CLIP_TOP_K,
        "use_clip": bool(use_clip),
        "use_image_prior": bool(use_image_prior),
        "low_info_penalty": RETRIEVAL_LOW_INFO_PENALTY if use_image_prior else 0.0,
        "low_info_in_pool": n_low_info,
        "min_evidence": RETRIEVAL_MIN_EVIDENCE if apply_gate else None,
        "use_sparse": use_sparse,
        "use_df_filter": use_df_filter,
        "apply_gate": apply_gate,
        "pool_size": len(pool),
        "bm25_max": round(max_bm25, 3),
        "bm25_ref_percentile": RETRIEVAL_BM25_REF_PERCENTILE,
        "evidence_range": [round(min(evs), 4), round(max(evs), 4)],
        "kept": len(items),
    }
    trace["effective_query"] = effective
    trace["total_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    return RetrievalResult(query, normalized, used, items, gated, trace)


def build_context(items: list[RetrievedChunk], max_chars: int = 6000) -> str:
    """
    把召回块拼成给 LLM 的上下文。每块标注来源页码，便于模型引用与人工核对。
    """
    parts: list[str] = []
    total = 0
    for i, it in enumerate(items, 1):
        header = f"[片段{i}] 来源：{it.doc} 第{it.page}页"
        if it.page_end and it.page_end != it.page:
            header = f"[片段{i}] 来源：{it.doc} 第{it.page}-{it.page_end}页"
        if it.section:
            header += f"（{it.section}）"
        if it.type == "table":
            header += "【表格】"
        elif it.type == "image":
            header += "【图片·多模态解析】"
        block = f"{header}\n{it.text}"
        if total + len(block) > max_chars and parts:
            break
        parts.append(block)
        total += len(block)
    return "\n\n".join(parts)
