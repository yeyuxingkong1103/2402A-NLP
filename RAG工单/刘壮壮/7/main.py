# -*- coding: utf-8 -*-
"""
main.py — RAG 功能测试及评估
工单编号：人工智能NLP-RAG-功能测试及评估

基于工单 01-06 实现的 RAG，对附件 ccf_competition.zip 中的金融年报 PDF
做 RAG 测试与检索评估：
  - 用 10 个问题做检索与生成
  - 用 RAGAS 风格指标（上下文精度/召回/忠实度）评估检索结果
  - 分析检索结果存在的问题

运行：
  python main.py                # 全量测试 + 评估 + 问题分析
  python main.py --query "xxx"  # 单问题
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rag_core import (Config, VectorStore, BM25Index, split_text, make_embedder,
                      make_llm, answer, build_context, evaluate_rag,
                      try_ragas_eval, HybridRetriever)  # noqa: E402
from questions import QUESTIONS  # noqa: E402

ATTACH_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "附件")
# ccf 数据已附带 txt（年报正文提取结果），直接使用 txt 更快且稳定
TXT_DIR = os.path.join(ATTACH_DIR, "ccf_competition", "txt")


def load_documents(txt_dir: str) -> list:
    """加载 ccf 年报 txt，按文档分块。返回 [(doc_name, chunk), ...]"""
    files = sorted(glob.glob(os.path.join(txt_dir, "*.txt")))
    docs = []
    for f in files:
        name = os.path.basename(f)
        with open(f, "r", encoding="utf-8", errors="ignore") as fp:
            text = fp.read()
        for c in split_text(text, 400, 60):
            docs.append((name, c))
    return docs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, default=None)
    args = parser.parse_args()

    config = Config()
    llm = make_llm(config)

    print(f"[加载] {TXT_DIR}")
    docs = load_documents(TXT_DIR)
    print(f"  共 {len(docs)} 个 chunk，来自 {len(set(d[0] for d in docs))} 份年报")

    chunks = [f"[{name}] {text}" for name, text in docs]
    embedder = make_embedder(config.embedding_model, chunks)
    vs = VectorStore(embedder, chunks)
    bm25 = BM25Index(chunks)
    retriever = HybridRetriever(vs, bm25)

    if args.query:
        hits = retriever.retrieve(args.query, top_k=5)
        print(f"\n问题：{args.query}")
        print(f"回答：{answer(args.query, hits, llm)}")
        for h in hits[:3]:
            print(f"  ({h['score']:.3f}) {h['text'][:80]}")
        return

    print("\n" + "=" * 72)
    print("RAG 测试 + 评估结果")
    print("=" * 72)

    rows = []
    for q in QUESTIONS:
        hits = retriever.retrieve(q["question"], top_k=5)
        ans = answer(q["question"], hits, llm)
        ctx = build_context(hits, 4)
        m = evaluate_rag(q["question"], q["reference"], ctx, ans)
        print(f"\n【问题 {q['id']}】{q['question']}")
        print(f"  [答案] {ans[:100]}")
        print(f"  [命中来源] {[h['text'][:h['text'].find(']')+1] if ']' in h['text'][:30] else '?' for h in hits[:3]]}")
        print(f"  [指标] 上下文精度={m['context_precision']:.3f} 召回={m['context_recall']:.3f} 忠实度={m['faithfulness']:.3f}")
        rows.append({
            "question": q["question"], "answer": ans,
            "contexts": [h["text"] for h in hits[:4]], "reference": q["reference"],
            "metrics": m,
        })

    # RAGAS 真实评估（若安装）
    ragas_result = try_ragas_eval(rows)
    if ragas_result:
        print("\n[RAGAS 评估结果]")
        for k, v in ragas_result.items():
            print(f"  {k}: {v}")

    # 检索问题分析
    print("\n" + "=" * 72)
    print("检索结果存在的问题分析")
    print("=" * 72)
    analyze_problems(rows)


def analyze_problems(rows):
    """分析检索结果存在的问题（低分项归因）。"""
    low_cp = [r for r in rows if r["metrics"]["context_precision"] < 0.5]
    low_cr = [r for r in rows if r["metrics"]["context_recall"] < 0.3]
    print(f"1. 上下文精度偏低的问题：{len(low_cp)} 个")
    for r in low_cp:
        print(f"   - 问题{r['question'][:30]}...（精度 {r['metrics']['context_precision']:.3f}）")
    print(f"2. 上下文召回偏低的问题：{len(low_cr)} 个")
    for r in low_cr:
        print(f"   - 问题{r['question'][:30]}...（召回 {r['metrics']['context_recall']:.3f}）")
    print("""\n可能原因与改进方向：
   - 离线 TF-IDF 向量缺乏语义泛化，同义表述（如"营收"vs"收入"）召回差 → 换 bge/m3e 语义向量
   - 分块粒度固定 400 字，跨段落的答案被切断 → 增大 chunk_size / 引入父子分块
   - 纯向量检索漏掉精确关键词 → 已用混合检索（向量+BM25）缓解
   - 缺少重排，TopK 内噪音干扰 → 加入 LLM/TF-IDF 重排
   - 参考答案涉及跨文档综合（多份年报对比） → 需多文档检索与聚合""")


if __name__ == "__main__":
    main()
