"""
检索模块（稠密 + 稀疏 加权融合）
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化

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
    RETRIEVAL_BM25_BETA,
    RETRIEVAL_DENSE_TOP_K,
    RETRIEVAL_DOC_CONSENSUS_DELTA,
    RETRIEVAL_DOC_HINT_DELTA,
    RETRIEVAL_FINAL_TOP_K,
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
    cosine: float = 0.0
    bm25_raw: float = 0.0
    bm25_norm: float = 0.0
    evidence: float = 0.0   # 依据分：余弦 + β·BM25归一（不含主题先验）
    consensus: float = 0.0  # 文档共识先验 0~1
    score: float = 0.0      # 融合分：依据分 + δ·共识（**可以 > 1，不是相似度**）

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
            "score": round(self.score, 4),
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

    # ---- 4) 加权融合
    max_bm25 = float(sparse.max()) if (use_sparse and sparse.size) else 0.0
    beta = RETRIEVAL_BM25_BETA if use_sparse else 0.0
    items: list[RetrievedChunk] = []
    for i in pool:
        c = kb.chunks[i]
        cos = float(dense[i])
        b_raw = float(sparse[i])
        b_norm = (b_raw / max_bm25) if max_bm25 > 0 else 0.0
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
                cosine=cos,
                bm25_raw=b_raw,
                bm25_norm=b_norm,
                evidence=ev,
            )
        )

    # ---- 5) 文档级共识先验 + 文档消歧加成（都是加法，都只影响排序）
    # 统计候选池里每个来源文档命中的块数；单文档语料下所有块共识相同，
    # 相当于加一个常数，不改变排序，也不会误伤闸门（闸门只看依据分）。
    doc_counts: dict[str, int] = {}
    for it in items:
        doc_counts[it.doc] = doc_counts.get(it.doc, 0) + 1
    max_doc_hit = max(doc_counts.values()) if doc_counts else 1
    for it in items:
        it.consensus = doc_counts.get(it.doc, 0) / max_doc_hit
        bonus = hint_bonus if (hint and it.doc_key in hint) else 0.0
        it.score = it.evidence + RETRIEVAL_DOC_CONSENSUS_DELTA * it.consensus + bonus

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
        "min_evidence": RETRIEVAL_MIN_EVIDENCE if apply_gate else None,
        "use_sparse": use_sparse,
        "use_df_filter": use_df_filter,
        "apply_gate": apply_gate,
        "pool_size": len(pool),
        "bm25_max": round(max_bm25, 3),
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
        block = f"{header}\n{it.text}"
        if total + len(block) > max_chars and parts:
            break
        parts.append(block)
        total += len(block)
    return "\n\n".join(parts)
