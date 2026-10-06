# -*- coding: utf-8 -*-
"""
工单06 基于用户反馈的自适应重排演示（越用越准）
工单编号：人工智能NLP-RAG-混合检索任务

演示内容：
  1. 同一 query 在「反馈前 / 反馈后」的 Top-10 排序对比；
  2. 模拟用户对检索结果点赞/点踩（3 轮反馈），展示被点赞片段排名上升、
     被点踩片段排名下降；
  3. 反馈迁移效应：同一批反馈应用到「相似问法」的新 query 上，排序同样改善；
  4. 技术说明：反馈特征 -> sigmoid 加权 -> 融合原检索分数的在线学习机制。

运行：
    python 工单06-混合检索/src/adaptive_feedback_demo.py
产出：
    工单06-混合检索/results/adaptive_demo.md

说明：本演示使用独立的反馈文件（results/adaptive_demo_feedback.json），
      不会污染线上反馈 data/index/rerank_feedback.json。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rag_core import rerank as rerank_mod                    # noqa: E402
from wo06_common import (RECALL_K, RESULTS_DIR, anchor_hit,  # noqa: E402
                         get_question, md_table, save_markdown, snip)
from build_index import DEFAULT_MODEL, ensure_index, get_retriever  # noqa: E402

QID = 260                                    # 主演示问题（军用领域收入）
PARAPHRASE = "兴图新科的军品业务收入大概有多少？"   # 相似问法（反馈迁移演示）
TOPK_DISPLAY = 10
ROUNDS = 3                                   # 模拟 3 位用户给出反馈


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def fresh_reranker(with_feedback: bool):
    """
    创建自适应重排器。
    with_feedback=False 时临时指向一个不存在的反馈文件，表示「反馈前」状态。
    """
    from rag_core.rerank import AdaptiveReranker

    if with_feedback:
        return AdaptiveReranker()
    original = rerank_mod.FEEDBACK_FILE
    rerank_mod.FEEDBACK_FILE = RESULTS_DIR / "_no_feedback.json"   # 不存在 -> 空反馈
    try:
        return AdaptiveReranker()
    finally:
        rerank_mod.FEEDBACK_FILE = original


def rank_map(docs: list[dict]) -> dict[str, int]:
    return {d["chunk_id"]: i for i, d in enumerate(docs, 1)}


def rank_table(before: list[dict], after: list[dict], qid: int) -> str:
    """反馈前/后排名对比表（以反馈后顺序为主）。"""
    q = get_question(qid)
    rb, ra = rank_map(before), rank_map(after)
    all_docs = {d["chunk_id"]: d for d in before + after}
    rows = []
    for cid, d in sorted(all_docs.items(), key=lambda kv: ra.get(kv[0], 999)):
        rank_b, rank_a = rb.get(cid, "—"), ra.get(cid, "—")
        arrow = ""
        if isinstance(rank_b, int) and isinstance(rank_a, int):
            arrow = "↑ 上升" if rank_a < rank_b else ("↓ 下降" if rank_a > rank_b else "持平")
        anchors = "、".join(a for a in (q.anchors if q else []) if anchor_hit(d, a)) or "—"
        rows.append([f"《{d.get('doc', '')}》p{d.get('page', '')}", snip(cid, 22),
                     anchors, rank_b, rank_a, arrow, snip(d.get("text", ""), 46)])
    return md_table(["来源", "chunk_id", "命中关键信息点", "反馈前排名", "反馈后排名",
                     "变化", "片段摘要"], rows)


def candidates(retr, query: str) -> list[dict]:
    """取融合检索候选池（不重排），作为固定输入观察重排效果。"""
    res = retr.retrieve(query, strategy="hybrid", fusion="rrf",
                        reranker="none", top_k=RECALL_K, recall_k=RECALL_K)
    return res.docs


def pick_feedback_targets(docs: list[dict], qid: int) -> tuple[dict, dict]:
    """
    自动挑选「点赞 / 点踩」目标，模拟真实用户行为：
      · 点赞：Top-10 中命中关键信息点最多的片段（对回答最有用）
      · 点踩：优先选来自错误文档的片段；否则选排名最低的片段
    """
    q = get_question(qid)
    top = docs[:TOPK_DISPLAY] or docs
    liked = max(top, key=lambda d: sum(anchor_hit(d, a) for a in q.anchors)) if q else top[0]
    disliked = next((d for d in top if d["chunk_id"] != liked["chunk_id"]
                     and q and q.doc not in (d.get("doc") or "")), None)
    if disliked is None:
        disliked = next((d for d in reversed(top) if d["chunk_id"] != liked["chunk_id"]), liked)
    return liked, disliked


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main() -> None:
    feedback_file = RESULTS_DIR / "adaptive_demo_feedback.json"
    if feedback_file.exists():
        feedback_file.unlink()
    original_feedback = rerank_mod.FEEDBACK_FILE
    rerank_mod.FEEDBACK_FILE = feedback_file

    try:
        from rag_core.rerank import AdaptiveReranker

        ensure_index(verbose=True)
        retr = get_retriever(DEFAULT_MODEL)
        main_q = get_question(QID)
        query = main_q.question

        print("=" * 72)
        print("工单06 自适应重排演示 | 用户反馈驱动排序优化")
        print(f"演示 query：{query}")
        print("=" * 72)

        # ---- 1. 反馈前 ----
        cand = candidates(retr, query)
        before = fresh_reranker(with_feedback=False).rerank(query, cand, top_k=TOPK_DISPLAY)
        print("\n[反馈前] Top-5：")
        for i, d in enumerate(before[:5], 1):
            print(f"  {i}. 《{d['doc']}》p{d['page']} "
                  f"final={float(d['final_score']):.4f} | {snip(d['text'], 52)}")

        # ---- 2. 模拟用户反馈 ----
        liked, disliked = pick_feedback_targets(cand, QID)
        actions = []
        rr = AdaptiveReranker()
        for _ in range(ROUNDS):
            rr.record(query, liked, helpful=True)
            rr.record(query, disliked, helpful=False)
        actions.append(f"点赞 {ROUNDS} 次：`{liked['chunk_id']}`"
                       f"（《{liked['doc']}》p{liked['page']}，"
                       f"{snip(liked['text'], 40)}）")
        actions.append(f"点踩 {ROUNDS} 次：`{disliked['chunk_id']}`"
                       f"（《{disliked['doc']}》p{disliked['page']}，"
                       f"{snip(disliked['text'], 40)}）")
        print(f"\n[模拟反馈] 点赞 {liked['chunk_id']}，点踩 {disliked['chunk_id']}"
              f"（各 {ROUNDS} 次）")
        print(f"[反馈统计] {rr.stats()}")

        # ---- 3. 反馈后 ----
        after = rr.rerank(query, cand, top_k=TOPK_DISPLAY)
        print("\n[反馈后] Top-5：")
        for i, d in enumerate(after[:5], 1):
            print(f"  {i}. 《{d['doc']}》p{d['page']} "
                  f"final={float(d['final_score']):.4f} | {snip(d['text'], 52)}")
        rb, ra = rank_map(before), rank_map(after)
        print(f"\n被点赞片段排名：{rb.get(liked['chunk_id'], '—')} -> "
              f"{ra.get(liked['chunk_id'], '—')}")
        print(f"被点踩片段排名：{rb.get(disliked['chunk_id'], '—')} -> "
              f"{ra.get(disliked['chunk_id'], '—')}")

        # ---- 4. 反馈迁移：相似问法 ----
        p_cand = candidates(retr, PARAPHRASE)
        p_before = fresh_reranker(with_feedback=False).rerank(
            PARAPHRASE, p_cand, top_k=TOPK_DISPLAY)
        p_after = AdaptiveReranker().rerank(PARAPHRASE, p_cand, top_k=TOPK_DISPLAY)
        p_rb, p_ra = rank_map(p_before), rank_map(p_after)
        moved = sum(1 for cid in p_ra if p_rb.get(cid) != p_ra[cid])
        print(f"\n[反馈迁移] 相似问法 Top-10 中有 {moved} 条排名变化")

        # ---- 5. 报告 ----
        q = get_question(QID)
        blocks = [
            "本报告由 `src/adaptive_feedback_demo.py` 自动生成，"
            "使用隔离的反馈文件，真实执行「反馈写入 -> 重排」全过程。\n\n"
            f"- 演示问题：{query}\n"
            f"- 检索策略：hybrid + rrf（候选池 Top-{RECALL_K}），自适应重排输出 Top-{TOPK_DISPLAY}\n"
            f"- 关键信息点：{'、'.join(q.anchors) if q else '—'}\n"
            f"- 模拟反馈：{ROUNDS} 轮点赞/点踩\n",
            "## 一、模拟的用户行为\n\n" + "\n".join(f"- {a}" for a in actions) +
            f"\n\n反馈统计：{rr.stats()}",
            "## 二、同一 query 反馈前 / 反馈后排序对比\n\n"
            + rank_table(before, after, QID) +
            f"\n\n**结论**：被点赞片段排名 "
            f"{rb.get(liked['chunk_id'], '—')} -> {ra.get(liked['chunk_id'], '—')}，"
            f"被点踩片段排名 {rb.get(disliked['chunk_id'], '—')} -> "
            f"{ra.get(disliked['chunk_id'], '—')}。"
            "自适应重排器把用户认可的信号沉淀为词条权重，"
            "在不重新训练模型的前提下即时改变排序。",
            f"## 三、反馈迁移（相似问法）\n\n"
            f"新 query：`{PARAPHRASE}`（与演示问题语义相近）\n\n"
            f"反馈前后 Top-{TOPK_DISPLAY} 中有 **{moved}** 条排名发生变化。\n\n"
            + rank_table(p_before, p_after, QID),
            """## 四、技术说明（自适应重排原理）

**在线学习**：每次点赞/点踩，把 query 与片段分词后的词条累计反馈分：

```
term_feedback[t] += +1（点赞） / -1（点踩）
type_feedback[片段类型] += +1 / -1
```

**重排打分**：查询词条的反馈总分经 sigmoid 压缩到 (0,1)，与原检索分数加权：

```
prior(d)  = Σ_{t∈query} term_feedback[t] + type_feedback[type_d]
adapted(d)= 1 / (1 + exp(-prior/5))        # 压缩，防止少量极端反馈带偏
final(d)  = 0.7 · base_norm(d) + 0.3 · adapted(d)
```

**特点**：无需重训模型、无网络开销，反馈即生效；相似问法共享词条级反馈，
因此一次点赞/点踩可以迁移到同主题的新问题，实现「越用越准」。""",
        ]

        path = save_markdown(RESULTS_DIR / "adaptive_demo.md",
                             "工单06 用户反馈自适应重排演示报告", blocks)
        print(f"\n[完成] 报告已输出 -> {path}")
    finally:
        rerank_mod.FEEDBACK_FILE = original_feedback


if __name__ == "__main__":
    main()
