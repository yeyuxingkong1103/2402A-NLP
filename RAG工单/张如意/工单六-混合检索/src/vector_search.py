# -*- coding: utf-8 -*-
"""
工单06 向量检索演示（多嵌入模型 + 召回 + 重排）
工单编号：人工智能NLP-RAG-混合检索任务

本脚本演示工单「向量检索（召回+重排）」的全部功能点：
  1. 多嵌入模型切换对比 —— bge-large-zh-v1.5（1024 维） vs m3e-base（768 维）
     · 同一 query 的 Top-5 召回对比、两模型结果重合度、检索耗时
     · 含一条英文 query，观察多语言检索表现（工单验收「多语言支持」）
  2. 三种重排算法对比 —— LLM 重排 / TF-IDF 重排 / 自适应重排（+ none 基线）
     · 向量召回 Top-20 -> 重排 -> Top-5，对比排序变化、分数与耗时
  3. 技术说明 —— BGE 嵌入 + 余弦相似度 + HNSW 近似最近邻

运行：
    python 工单06-混合检索/src/vector_search.py
产出：
    工单06-混合检索/results/vector_demo.md

说明：LLM 重排需要 DeepSeek API（未配置 DEEPSEEK_API_KEY 时自动回退，
      不影响脚本运行）；首次运行需从 HuggingFace 下载嵌入模型。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rag_core import config                                   # noqa: E402
from wo06_common import (RECALL_K, RESULTS_DIR, TOP_K, md_table,  # noqa: E402
                         question_passed, save_markdown, snip)
from build_index import COLLECTIONS, DEFAULT_MODEL, ensure_index, get_retriever  # noqa: E402

# 演示用 query（前 3 条为工单问题集中的问题，最后 1 条为英文问法）
MODEL_QUERIES = [
    (260, "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？"),
    (543, "武汉兴图新科电子股份有限公司注册资本是多少？"),
    (795, "武汉兴图新科电子股份有限公司参与的哪个工程荣获了国家科技进步一等奖？"),
    (None, "What is the registered capital of Wuhan Xingtu Xinke Electronics?"),
]

RERANK_QUERIES = [
    (260, "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？"),
    (207, "武汉兴图新科电子股份有限公司计划使用本次发行募集资金的多少用于补充流动资金？"),
]


def hits_table(docs: list[dict], top: int = 5) -> str:
    rows = []
    for i, d in enumerate(docs[:top], 1):
        score = float(d.get("final_score", d.get("score", 0)))
        rows.append([i, f"《{d.get('doc', '')}》", d.get("page", ""),
                     d.get("type", "text"), f"{score:.4f}",
                     snip(d.get("text", ""), 72)])
    return md_table(["排名", "文档", "页码", "类型", "分数", "片段摘要"], rows) \
        if rows else "（无命中）"


# ---------------------------------------------------------------------------
# 1. 多嵌入模型召回对比
# ---------------------------------------------------------------------------
def demo_models() -> list[str]:
    lines = ["## 1. 多嵌入模型召回对比",
             "同一问题、同一向量库规模，仅切换嵌入模型；分数为归一化向量的余弦相似度。"]
    per_model: dict[str, dict[int | str, list[dict]]] = {}

    for model, coll in COLLECTIONS.items():
        try:
            retr = get_retriever(model, verbose=True)
        except Exception as e:                        # 模型下载失败不影响其他演示
            lines.append(f"### 模型 {model}\n\n> 跳过：{e}")
            print(f"[warn] 模型 {model} 不可用：{e}")
            continue

        per_model[model] = {}
        print(f"\n{'=' * 72}\n嵌入模型：{model}（集合 {coll}）\n{'=' * 72}")
        lines.append(f"### 模型 {model}（向量库集合 `{coll}`）")

        for qid, q in MODEL_QUERIES:
            t0 = time.perf_counter()
            docs = retr.vector_search(q, recall_k=RECALL_K)[:TOP_K]
            cost = time.perf_counter() - t0
            per_model[model][qid if qid else q] = docs
            tag = f"问题{qid}" if qid else "英文问题"
            print(f"\n[{tag}] {q}  (Top-{TOP_K}, {cost * 1000:.0f} ms)")
            for i, d in enumerate(docs, 1):
                print(f"  {i}. 《{d.get('doc')}》第{d.get('page')}页 "
                      f"cos={float(d.get('score', 0)):.4f} | {snip(d.get('text', ''), 58)}")
            lines.append(f"\n**{tag}**：{q}\n\n检索耗时 {cost * 1000:.0f} ms\n\n"
                         + hits_table(docs))

    # 两模型 Top-5 重合度（衡量模型差异，供模型选型参考）
    models = list(per_model)
    if len(models) == 2:
        m1, m2 = models
        rows = []
        for key in per_model[m1]:
            ids1 = {d["chunk_id"] for d in per_model[m1][key]}
            ids2 = {d["chunk_id"] for d in per_model[m2][key]}
            inter = len(ids1 & ids2)
            rows.append([str(key)[:26], inter, f"{inter / max(len(ids1 | ids2), 1):.0%}"])
        lines.append("### 两模型 Top-5 结果重合度\n\n"
                     + md_table(["查询", "重合片段数", "Jaccard 重合度"], rows) +
                     "\n\n> 重合度低说明两个模型召回的侧重点不同；"
                     "混合检索/多路召回时可取长补短。")
    return lines


# ---------------------------------------------------------------------------
# 2. 三种重排算法对比
# ---------------------------------------------------------------------------
def demo_rerankers() -> list[str]:
    from rag_core.rerank import AdaptiveReranker, get_reranker

    lines = ["## 2. 三种重排算法对比（向量召回 + 重排）",
             f"先向量召回 Top-{RECALL_K}（recall_k），再用不同重排器输出 Top-{TOP_K}。",
             "若未配置 DEEPSEEK_API_KEY，LLM 重排会自动回退为原顺序（打印 warn）。"]
    if not config.DEEPSEEK_API_KEY:
        lines.append("> 注意：当前未检测到 `DEEPSEEK_API_KEY`，"
                     "LLM 重排与级联重排的分数仅供参考，配置后可复跑本脚本。")

    retr = get_retriever(DEFAULT_MODEL)
    rerankers = ["none", "tfidf", "llm", "adaptive"]

    for qid, q in RERANK_QUERIES:
        # 向量召回（不重排），作为各重排器的同一输入
        base = retr.retrieve(q, strategy="vector", top_k=RECALL_K,
                             recall_k=RECALL_K, reranker="none")
        base_docs = base.docs
        print(f"\n{'=' * 72}\n问题{qid}：{q}\n{'=' * 72}")
        lines.append(f"### 问题{qid}：{q}\n\n"
                     f"向量召回 Top-{RECALL_K}，耗时 {base.timings.get('vector_recall', 0) * 1000:.0f} ms")

        for name in rerankers:
            t0 = time.perf_counter()
            if name == "adaptive":
                rr = AdaptiveReranker()          # 读取已积累的用户反馈（可能为空）
            else:
                rr = get_reranker(name)
            docs = rr.rerank(q, [dict(d) for d in base_docs], top_k=TOP_K) \
                if rr is not None else base_docs[:TOP_K]
            cost = time.perf_counter() - t0

            base_rank = {d["chunk_id"]: i for i, d in enumerate(base_docs, 1)}
            moved = sum(1 for i, d in enumerate(docs, 1)
                        if base_rank.get(d["chunk_id"], -1) != i)
            print(f"\n[重排器 {name}] 耗时 {cost * 1000:.0f} ms，"
                  f"Top-{TOP_K} 中有 {moved} 条偏离原始召回位次")
            for i, d in enumerate(docs, 1):
                print(f"  {i}. 《{d.get('doc')}》第{d.get('page')}页 "
                      f"final={float(d.get('final_score', d.get('score', 0))):.4f} | "
                      f"{snip(d.get('text', ''), 56)}")
            lines.append(f"#### 重排器 `{name}`（耗时 {cost * 1000:.0f} ms，"
                         f"{moved} 条偏离原始位次）\n\n" + hits_table(docs))

        # 定量：不同重排器下 Top-5 是否覆盖全部关键信息点
        from wo06_common import WO06_QA_SET
        qobj = next((x for x in WO06_QA_SET if x.qid == qid), None)
        if qobj:
            rows = []
            for name in rerankers:
                if name == "adaptive":
                    rr = AdaptiveReranker()
                else:
                    rr = get_reranker(name)
                docs = rr.rerank(q, [dict(d) for d in base_docs], top_k=TOP_K) \
                    if rr is not None else base_docs[:TOP_K]
                rows.append([name, "是" if question_passed(docs, qobj) else "否",
                             snip("、".join(qobj.anchors), 40)])
            lines.append("**关键信息点覆盖检查**（该问题能否直接答对）\n\n"
                         + md_table(["重排器", "Top-5 覆盖全部关键信息点", "关键信息点"], rows))
    return lines


def main() -> None:
    # 只保证默认模型就绪；m3e 在 demo_models 内按模型尝试（失败自动跳过）
    ensure_index([DEFAULT_MODEL], verbose=True)
    print("=" * 72)
    print("工单06 向量检索演示 | BGE 嵌入 + 余弦相似度 + HNSW")
    print("=" * 72)

    blocks = [
        "本报告由 `src/vector_search.py` 自动生成，所有结果均为真实检索所得。\n\n"
        "向量检索链路：文本 -> 嵌入模型（BGE/M3E）-> L2 归一化向量 -> "
        "HNSW 近似最近邻召回（余弦相似度）-> 重排器精排。",
    ]
    blocks += demo_models()
    blocks += demo_rerankers()
    blocks.append("""## 3. 技术说明

**语义检索技术栈**：
1. 嵌入模型：bge-large-zh-v1.5（1024 维）/ m3e-base（768 维），
   查询侧自动加 BGE 官方检索指令前缀，文档侧不加（`rag_core.embed.encode`）。
2. 相似度：向量 L2 归一化后，余弦相似度 = 向量点积：
   `cos(q,d) = q·d / (|q||d|) = q̂·d̂`（归一化后 |q̂|=|d̂|=1）。
3. 近邻索引：ChromaDB 持久化集合，`hnsw:space=cosine`，
   HNSW 通过多层可导航小世界图把 Top-k 复杂度从 O(N) 降到近似 O(log N)。
4. 重排：召回 Top-20 后由 LLM / TF-IDF / 自适应重排器精排 Top-5，
   兼顾召回率（粗排）与准确率（精排）。""")

    path = save_markdown(RESULTS_DIR / "vector_demo.md",
                         "工单06 向量检索演示报告（多嵌入模型 + 重排）", blocks)
    print(f"\n[完成] 报告已输出 -> {path}")


if __name__ == "__main__":
    main()
