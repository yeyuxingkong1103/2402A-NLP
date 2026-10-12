# -*- coding: utf-8 -*-
"""
main.py — LightRAG 优化
工单编号：人工智能NLP-RAG 项目-LightRAG 优化

在 RAG 基础上，使用 LightRAG 对《招股说明书1.pdf》《招股说明书2.pdf》
做知识图谱构建与检索：
  - 优化图谱构建中的实体、关系，抽取准确的实体类型和关系类型
  - 检索时可选择 RAG 或 LightRAG，比对二者对测试问题的检索结果
  - 用 RAGAS 指标整理 RAG / LightRAG 对比

运行：
  python main.py                       # 对比 RAG vs LightRAG + RAGAS 指标
  python main.py --query "xxx"         # 单问题（LightRAG）
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rag_core import (Config, VectorStore, BM25Index, load_pdf_text, split_text,
                      make_embedder, make_llm, answer, build_context,
                      HybridRetriever, try_ragas_eval)  # noqa: E402
from lightrag import LightRAG  # noqa: E402
from questions import QUESTIONS  # noqa: E402

ATTACH_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "附件")
PDF1 = os.path.join(ATTACH_DIR, "招股说明书1-无水印.pdf")
if not os.path.exists(PDF1):
    PDF1 = os.path.join(ATTACH_DIR, "招股说明书1.pdf")
PDF2 = os.path.join(ATTACH_DIR, "招股说明书2.pdf")


def load_chunks(paths):
    chunks = []
    for p in paths:
        if os.path.exists(p):
            text = load_pdf_text(p)
            chunks.extend(split_text(text, 400, 60))
    return chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, default=None)
    args = parser.parse_args()

    config = Config()
    llm = make_llm(config)
    chunks = load_chunks([PDF1, PDF2])
    print(f"[加载] {len(chunks)} 块（招股说明书1 + 2）")

    # 传统 RAG 索引
    print("[RAG] 构建向量 + BM25 索引")
    embedder = make_embedder(config.embedding_model, chunks)
    rag_retriever = HybridRetriever(VectorStore(embedder, chunks), BM25Index(chunks))

    # LightRAG 索引
    print("[LightRAG] 构建知识图谱 + 双层检索索引")
    lrag = LightRAG(llm)
    lrag.build(chunks)
    s = lrag.stats()
    print(f"  图谱：节点 {s['nodes']}，边 {s['edges']}，实体 {s['entities']}，主题 {s['topics']}")

    if args.query:
        lhits = lrag.retrieve(args.query, top_k=5)
        rhits = rag_retriever.retrieve(args.query, top_k=5)
        print(f"\n问题：{args.query}")
        print(f"[LightRAG 回答] {answer(args.query, lhits, llm)}")
        print(f"[RAG 回答]     {answer(args.query, rhits, llm)}")
        return

    print("\n" + "=" * 78)
    print(f"{'问题':<6}{'RAG命中':<12}{'LightRAG命中':<14}")
    print("=" * 78)
    rag_rows, lrag_rows = [], []
    for q in QUESTIONS:
        rh = rag_retriever.retrieve(q["question"], top_k=5)
        lh = lrag.retrieve(q["question"], top_k=5)
        ra = answer(q["question"], rh, llm)
        la = answer(q["question"], lh, llm)
        # 命中度：首条结果与问题的词面重叠（近似评估）
        print(f"{q['id']:<6}{rag_hit(q['question'], rh):<12.3f}{rag_hit(q['question'], lh):<14.3f}")
        rag_rows.append({"question": q["question"], "answer": ra,
                         "contexts": [h["text"] for h in rh[:4]], "reference": ""})
        lrag_rows.append({"question": q["question"], "answer": la,
                          "contexts": [h["text"] for h in lh[:4]], "reference": ""})
    print("=" * 78)

    # RAGAS 指标对比
    r_ragas = try_ragas_eval(rag_rows)
    l_ragas = try_ragas_eval(lrag_rows)
    if r_ragas and l_ragas:
        print("\nRAGAS 指标对比：")
        print(f"{'指标':<24}{'RAG':<12}{'LightRAG':<12}")
        for k in r_ragas:
            print(f"{k:<24}{r_ragas[k]:<12.4f}{l_ragas.get(k, 0):<12.4f}")
    else:
        print("\n（未安装 ragas；已用内置离线命中度完成对比。安装：pip install ragas）")


def rag_hit(question, hits):
    """首条命中与问题的词面重叠度（离线近似评分）。"""
    from rag_core import tokenize
    if not hits:
        return 0.0
    q = set(tokenize(question))
    top = set(tokenize(hits[0]["text"]))
    if not q:
        return 0.0
    return len(q & top) / len(q)


if __name__ == "__main__":
    main()
