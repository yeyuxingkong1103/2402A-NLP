# -*- coding: utf-8 -*-
"""
工单06 混合检索主程序（向量 + 全文，三种融合算法 + 权重调节）
工单编号：人工智能NLP-RAG-混合检索任务

本脚本演示工单「混合检索」的全部功能点：
  1. 双通道并行执行   —— 向量检索（语义召回） + BM25 全文检索（关键词召回）
  2. 三种融合算法对比 —— weighted 加权平均 / rrf 倒数排名融合 / vote 投票机制
  3. 权重 alpha 调节  —— alpha 从 0（纯全文）到 1（纯向量）扫描，观察排序变化
  4. 通道互补性分析   —— 两通道 Top-k 重合度，解释混合检索为何更准

运行：
    python 工单06-混合检索/src/hybrid_search.py
产出：
    工单06-混合检索/results/hybrid_demo.md
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rag_core import config                                   # noqa: E402
from wo06_common import (RECALL_K, RESULTS_DIR, TOP_K,  # noqa: E402
                         anchor_hit, get_question, md_table, save_markdown, snip)
from build_index import DEFAULT_MODEL, ensure_index, get_retriever  # noqa: E402

QUERIES = [260, 34, 543]          # 工单问题集中的问题 id
ALPHAS = [round(i / 10, 1) for i in range(11)]   # 0.0, 0.1, ..., 1.0


def hits_table(docs: list[dict], top: int = TOP_K, show_extra: bool = True) -> str:
    rows = []
    for i, d in enumerate(docs[:top], 1):
        score = float(d.get("final_score", d.get("score", 0)))
        extra = ""
        if show_extra:
            if d.get("ranks"):
                r = d["ranks"]
                extra = f"向量#{r.get('vector', '—')} / BM25#{r.get('bm25', '—')}"
            elif d.get("votes") is not None:
                extra = f"票数={d.get('votes')}"
        # 标注该片段命中了问题的哪些关键信息点，便于观察融合效果
        q = get_question(d["_qid"]) if d.get("_qid") else None
        if q:
            hit = [a for a in q.anchors if anchor_hit(d, a)]
            if hit:
                extra += ("，" if extra else "") + "含：" + "、".join(hit)
        rows.append([i, f"《{d.get('doc', '')}》", d.get("page", ""),
                     d.get("source", ""), f"{score:.4f}", extra,
                     snip(d.get("text", ""), 56)])
    return md_table(["排名", "文档", "页码", "来源通道", "融合分数", "通道排名/投票", "片段摘要"], rows) \
        if rows else "（无命中）"


def tag_qid(docs: list[dict], qid: int) -> list[dict]:
    """给片段打上问题 id 标记，便于表格里标注关键信息点命中情况。"""
    return [{**d, "_qid": qid} for d in docs]


def target_chunk(vec: list[dict], bm: list[dict], qid: int) -> str | None:
    """
    选出「关键信息点命中最多」的片段作为观察目标：
    融合排序能否把它顶上来，是混合检索有效性的直接证据。
    """
    q = get_question(qid)
    if not q:
        return None
    pool: dict[str, dict] = {}
    for d in vec + bm:
        pool.setdefault(d["chunk_id"], d)
    if not pool:
        return None
    return max(pool, key=lambda cid: sum(anchor_hit(pool[cid], a) for a in q.anchors))


def rank_of(docs: list[dict], chunk_id: str | None) -> str:
    if chunk_id is None:
        return "—"
    for i, d in enumerate(docs, 1):
        if d["chunk_id"] == chunk_id:
            return str(i)
    return f">{len(docs)}"


def main() -> None:
    ensure_index(verbose=True)
    retr = get_retriever(DEFAULT_MODEL)
    print("=" * 72)
    print("工单06 混合检索演示 | weighted / rrf / vote")
    print(f"默认权重 alpha={config.HYBRID_ALPHA}（向量权重），RRF_K={config.RRF_K}")
    print("=" * 72)

    blocks = [
        "本报告由 `src/hybrid_search.py` 自动生成，真实执行双通道检索与三种融合算法。\n\n"
        f"- 召回阶段：每通道 Top-{RECALL_K}，融合后取 Top-{TOP_K}\n"
        f"- 默认权重：alpha（向量权重）={config.HYBRID_ALPHA}，"
        f"全文权重=1-alpha\n"
        f"- RRF 常数 k={config.RRF_K}（Cormack 等 2009 年论文推荐 60）",
    ]

    for qid in QUERIES:
        q = get_question(qid)
        if q is None:
            continue
        query = q.question
        print(f"\n{'#' * 72}\n问题{qid}：{query}\n{'#' * 72}")
        blocks.append(f"## 问题{qid}：{query}\n\n"
                      f"关键信息点：{'、'.join(q.anchors)}")

        # ---- 1. 双通道原始召回 ----
        vec = retr.vector_search(query, top_k=RECALL_K)
        bm = retr.fulltext_search(query, top_k=RECALL_K)
        vec_ids, bm_ids = {d["chunk_id"] for d in vec}, {d["chunk_id"] for d in bm}
        inter = len(vec_ids & bm_ids)
        union = len(vec_ids | bm_ids) or 1
        print(f"\n[通道] 向量 {len(vec)} 条，BM25 {len(bm)} 条，"
              f"重合 {inter} 条（Jaccard={inter / union:.0%}）")
        blocks.append(
            f"**通道互补性**：向量 Top-{RECALL_K} 与 BM25 Top-{RECALL_K} 重合 "
            f"{inter} 条，Jaccard 重合度 {inter / union:.0%}——"
            f"两个通道各自召回了对方没有的片段，融合具备信息增量。\n\n"
            f"### 向量通道 Top-{TOP_K}（语义召回）\n\n"
            + hits_table(tag_qid(vec, qid), show_extra=False) +
            f"\n\n### BM25 通道 Top-{TOP_K}（关键词召回）\n\n"
            + hits_table(tag_qid(bm, qid), show_extra=False))

        # ---- 2. 三种融合算法 ----
        tgt = target_chunk(vec, bm, qid)
        rows = []
        for fusion, desc in [("weighted", f"加权平均 alpha={config.HYBRID_ALPHA}"),
                             ("rrf", "倒数排名融合 k=60"),
                             ("vote", "投票机制（双通道命中记 2 票）")]:
            res = retr.retrieve(query, strategy="hybrid", fusion=fusion,
                                reranker="none", top_k=RECALL_K, recall_k=RECALL_K)
            docs = res.docs
            rows.append([fusion, desc, rank_of(docs, tgt),
                         f"{res.timings.get('fusion', 0) * 1000:.1f}",
                         snip(docs[0].get("text", "") if docs else "", 40)])
            print(f"\n[融合 {fusion}] 目标片段排名={rank_of(docs, tgt)}，"
                  f"融合耗时 {res.timings.get('fusion', 0) * 1000:.1f} ms")
            blocks.append(f"### 融合算法 `{fusion}`（{desc}）\n\n"
                          f"关键信息点最全的片段排名：**{rank_of(docs, tgt)}**"
                          f"（共 {len(docs)} 条参与排序）\n\n"
                          + hits_table(tag_qid(docs, qid)))
        blocks.append("**三种融合对比**\n\n" + md_table(
            ["融合算法", "参数", "目标片段排名", "融合耗时(ms)", "Top-1 摘要"], rows))

        # ---- 3. alpha 权重扫描 ----
        sweep_rows = []
        print(f"\n[权重扫描] alpha: 0.0(纯全文) -> 1.0(纯向量)")
        for alpha in ALPHAS:
            res = retr.retrieve(query, strategy="hybrid", fusion="weighted",
                                alpha=alpha, reranker="none",
                                top_k=RECALL_K, recall_k=RECALL_K)
            docs = res.docs
            top3 = " / ".join(f"《{d['doc'][-1]}》p{d['page']}" for d in docs[:3])
            sweep_rows.append([alpha, rank_of(docs, tgt), top3,
                               snip(docs[0].get("text", "") if docs else "", 34)])
        blocks.append("### 权重 alpha 扫描（weighted 融合）\n\n"
                      + md_table(["alpha（向量权重）", "目标片段排名",
                                  "Top-3 来源", "Top-1 摘要"], sweep_rows) +
                      "\n\n> alpha=0 等价纯全文检索，alpha=1 等价纯向量检索；"
                      "中间值兼顾语义与关键词信号，工单默认 alpha="
                      f"{config.HYBRID_ALPHA}。")
        print(md_table(["alpha", "目标片段排名", "Top-3 来源", "Top-1 摘要"], sweep_rows))

    blocks.append("""## 技术说明（三种融合算法）

**weighted 加权平均**（量纲敏感，需先归一化）：
```
score(d) = α · norm(vec_score_d) + (1-α) · norm(bm25_score_d)
norm(x) = (x - min) / (max - min)          # 各通道内 Min-Max 归一化
```
适合：两路检索质量相当、需要连续可调的权重。

**rrf 倒数排名融合**（Cormack et al., 2009）：
```
score(d) = Σ_{r∈R(d)} 1 / (k + rank_r(d))，k=60
```
只看排名不看分数，天然免疫「余弦相似度 vs BM25 分数量纲差异」，最稳健，工业界标配。

**vote 投票机制**：
```
votes(d) = 命中通道数 ∈ {1, 2}；按 (votes, 归一化分数和) 降序
```
双通道都命中的片段获得 2 票优先，强调「共识结果」，适合高风险问答场景。""")

    path = save_markdown(RESULTS_DIR / "hybrid_demo.md",
                         "工单06 混合检索演示报告（三融合算法 + 权重扫描）", blocks)
    print(f"\n[完成] 报告已输出 -> {path}")


if __name__ == "__main__":
    main()
