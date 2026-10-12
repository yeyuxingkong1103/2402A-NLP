# -*- coding: utf-8 -*-
"""
main.py — 基于 PDF 文档的问答系统优化
工单编号：人工智能NLP-RAG-基于PDF 文档的问答系统优化

针对工单 01，从「PDF 解析处理、分块优化、检索优化」三个层面提升检索准确率：
  1. 分块优化：结构感知分块（按标题/段落切分）+ 合理 overlap，减少语义割裂
  2. 检索优化：向量检索 + BM25 全文检索的混合检索（RRF 融合）
  3. 重排优化：TF-IDF / LLM 重排器对召回结果二次排序

运行：
  python main.py              # 对比优化前后检索精确度
  python main.py --query "x"  # 单问题检索
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rag_core import (Config, BM25Index, VectorStore, load_pdf_text, split_text,
                      make_embedder, make_llm, answer, rerank_by_tfidf, build_context,
                      evaluate_rag, try_ragas_eval)  # noqa: E402
from questions import QUESTIONS  # noqa: E402

ATTACH_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "附件")
PDF_PATH = os.path.join(ATTACH_DIR, "招股说明书1-无水印.pdf")
if not os.path.exists(PDF_PATH):
    PDF_PATH = os.path.join(ATTACH_DIR, "招股说明书1.pdf")


# ----------------------------------------------------------------------------
# 分块优化：结构感知分块（按段落 / 标题切分，保留语义完整）
# ----------------------------------------------------------------------------
def split_by_structure(text: str, chunk_size: int = 500, overlap: int = 80) -> list:
    """先按标题/换行切成段落，再合并成不超过 chunk_size 的语义块。"""
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks, buf = [], ""
    for para in paragraphs:
        if len(buf) + len(para) > chunk_size and buf:
            chunks.append(buf)
            buf = para
        else:
            buf = f"{buf}\n{para}" if buf else para
    if buf:
        chunks.append(buf)
    # 二次滑动窗口增加 overlap
    if overlap <= 0:
        return chunks
    overlapped = []
    for i, c in enumerate(chunks):
        if i > 0:
            c = chunks[i - 1][-overlap:] + "\n" + c
        overlapped.append(c)
    return overlapped


def build_index(chunks, config, hybrid: bool):
    """构建向量库与 BM25 索引，返回 (vector_store, bm25)。"""
    embedder = make_embedder(config.embedding_model, chunks)
    vs = VectorStore(embedder, chunks)
    bm25 = BM25Index(chunks)
    return vs, bm25, embedder


def retrieve_baseline(vs, query, top_k=5):
    """基线：仅向量检索。"""
    return vs.search(query, top_k=top_k)


def retrieve_optimized(vs, bm25, query, top_k=5, use_rerank=True):
    """优化：混合检索（向量 + BM25 的 RRF 融合）+ 重排。"""
    vec = {h["index"]: h for h in vs.search(query, top_k=top_k * 3)}
    kw = {h["index"]: h for h in bm25.search(query, top_k=top_k * 3)}
    rrf = {}
    for rank, i in enumerate(vec):
        rrf[i] = rrf.get(i, 0.0) + 1.0 / (rank + 60)
    for rank, i in enumerate(kw):
        rrf[i] = rrf.get(i, 0.0) + 1.0 / (rank + 60)
    order = sorted(rrf.items(), key=lambda x: -x[1])[:top_k]
    hits = [dict(vec.get(i) or kw.get(i), rrf=s) for i, s in order]
    if use_rerank:
        hits = rerank_by_tfidf(query, hits)
    return hits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, default=None)
    args = parser.parse_args()

    config = Config()
    llm = make_llm(config)
    print(f"[解析] {PDF_PATH}")
    text = load_pdf_text(PDF_PATH)

    # 基线：朴素分块 + 纯向量检索
    base_chunks = split_text(text, 400, 60)
    base_vs, _, base_emb = build_index(base_chunks, config, hybrid=False)

    # 优化：结构感知分块 + 混合检索 + 重排
    opt_chunks = split_by_structure(text, 500, 80)
    opt_vs, opt_bm25, opt_emb = build_index(opt_chunks, config, hybrid=True)

    print(f"基线分块：{len(base_chunks)} 块（朴素滑动窗口）")
    print(f"优化分块：{len(opt_chunks)} 块（结构感知）\n")

    if args.query:
        q = args.query
        print(f"问题：{q}")
        for name, hits in [("基线", retrieve_baseline(base_vs, q)),
                            ("优化", retrieve_optimized(opt_vs, opt_bm25, q))]:
            print(f"\n[{name}检索]")
            for h in hits[:3]:
                print(f"  ({h['score']:.3f}) {h['text'][:90]}")
        return

    print("=" * 72)
    print(f"{'问题ID':<6}{'基线精度':<10}{'优化精度':<10}{'基线召回':<10}{'优化召回':<10}")
    print("=" * 72)
    base_sum = {"cp": 0.0, "cr": 0.0}
    opt_sum = {"cp": 0.0, "cr": 0.0}
    samples = []
    for q in QUESTIONS:
        b_hits = retrieve_baseline(base_vs, q["question"])
        o_hits = retrieve_optimized(opt_vs, opt_bm25, q["question"])
        b_ctx, o_ctx = build_context(b_hits, 4), build_context(o_hits, 4)
        b_ans, o_ans = answer(q["question"], b_hits, llm), answer(q["question"], o_hits, llm)
        b_m = evaluate_rag(q["question"], "", b_ctx, b_ans)
        o_m = evaluate_rag(q["question"], "", o_ctx, o_ans)
        base_sum["cp"] += b_m["context_precision"]
        base_sum["cr"] += b_m["context_recall"]
        opt_sum["cp"] += o_m["context_precision"]
        opt_sum["cr"] += o_m["context_recall"]
        print(f"{q['id']:<6}{b_m['context_precision']:<10.3f}{o_m['context_precision']:<10.3f}"
              f"{b_m['context_recall']:<10.3f}{o_m['context_recall']:<10.3f}")
        samples.append({
            "question": q["question"], "answer": o_ans,
            "contexts": [h["text"] for h in o_hits[:4]], "reference": "",
        })

    n = len(QUESTIONS)
    print("=" * 72)
    print(f"{'平均':<6}{base_sum['cp']/n:<10.3f}{opt_sum['cp']/n:<10.3f}"
          f"{base_sum['cr']/n:<10.3f}{opt_sum['cr']/n:<10.3f}")
    print(f"\n上下文精度提升：{(opt_sum['cp']-base_sum['cp'])/n:+.3f}")
    print(f"上下文召回提升：{(opt_sum['cr']-base_sum['cr'])/n:+.3f}")

    ragas_result = try_ragas_eval(samples)
    if ragas_result:
        print("\n[RAGAS 评估]")
        for k, v in ragas_result.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
