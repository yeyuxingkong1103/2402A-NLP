# -*- coding: utf-8 -*-
"""
main.py — 混合检索
工单编号：人工智能NLP-RAG-混合检索任务

提供三种检索策略的可配置应用：
  1. 向量检索（召回 + 重排）：向量嵌入 + 相似度召回，重排算法二次排序
  2. 全文检索：BM25 倒排索引，支持关键词/短语/模糊匹配、多字段
  3. 混合检索：向量 + 全文加权融合（RRF）

支持：
  - 多种嵌入模型（bge / m3e / 离线 TF-IDF，经 EMBEDDING_MODEL 配置）
  - 至少 3 种重排算法（LLM / TF-IDF / 用户反馈自适应）
  - 向量与全文检索的权重调整

运行：
  python main.py                                   # 演示三种检索策略
  python main.py --query "xxx" --strategy hybrid   # 指定策略单问
  python main.py --strategy vector --rerank llm    # 指定重排器
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rag_core import (Config, VectorStore, BM25Index, load_pdf_text, split_text,
                      make_embedder, make_llm, answer, build_context)  # noqa: E402
from rerankers import get_reranker  # noqa: E402
from questions import QUESTIONS  # noqa: E402

ATTACH_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "附件")
PDF_PATH = os.path.join(ATTACH_DIR, "招股说明书1-无水印.pdf")
if not os.path.exists(PDF_PATH):
    PDF_PATH = os.path.join(ATTACH_DIR, "招股说明书1.pdf")


class RetrievalEngine:
    """可配置检索引擎：向量 / 全文 / 混合 + 重排。"""

    def __init__(self, chunks, config: Config):
        self.chunks = chunks
        self.config = config
        embedder = make_embedder(config.embedding_model, chunks)
        self.vs = VectorStore(embedder, chunks)
        self.bm25 = BM25Index(chunks)
        self.llm = make_llm(config)

    def search(self, query: str, strategy: str = "hybrid", top_k: int = 5,
               alpha: float = 0.5, rerank: str = "tfidf") -> list:
        """strategy: vector / fulltext / hybrid"""
        if strategy == "vector":
            hits = self.vs.search(query, top_k=top_k * 3)
        elif strategy == "fulltext":
            hits = self.bm25.search(query, top_k=top_k * 3)
        else:  # hybrid
            hits = self._hybrid(query, top_k * 3, alpha)
        # 重排
        reranker = get_reranker(rerank, self.llm)
        hits = reranker.rerank(query, hits)
        return hits[:top_k]

    def _hybrid(self, query: str, top_k: int, alpha: float) -> list:
        vec = {h["index"]: h for h in self.vs.search(query, top_k=top_k)}
        kw = {h["index"]: h for h in self.bm25.search(query, top_k=top_k)}
        rrf: dict = {}
        for rank, i in enumerate(vec):
            rrf[i] = rrf.get(i, 0.0) + alpha / (rank + 60)
        for rank, i in enumerate(kw):
            rrf[i] = rrf.get(i, 0.0) + (1 - alpha) / (rank + 60)
        order = sorted(rrf.items(), key=lambda x: -x[1])[:top_k]
        return [dict(vec.get(i) or kw.get(i), rrf=s) for i, s in order]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, default=None)
    parser.add_argument("--strategy", choices=["vector", "fulltext", "hybrid"], default="hybrid")
    parser.add_argument("--rerank", choices=["tfidf", "llm", "adaptive"], default="tfidf")
    parser.add_argument("--alpha", type=float, default=0.5, help="向量检索权重（0~1）")
    args = parser.parse_args()

    config = Config()
    print(f"[解析] {PDF_PATH}")
    text = load_pdf_text(PDF_PATH)
    chunks = split_text(text, config.chunk_size, config.chunk_overlap)
    engine = RetrievalEngine(chunks, config)

    if args.query:
        hits = engine.search(args.query, args.strategy, alpha=args.alpha, rerank=args.rerank)
        print(f"\n问题：{args.query}")
        print(f"策略：{args.strategy}，重排：{args.rerank}")
        print(f"回答：{answer(args.query, hits, engine.llm)}")
        print("\n检索结果：")
        for h in hits:
            print(f"  ({h['score']:.3f}) {h['text'][:80]}")
        return

    # 演示三种策略对比
    print("\n" + "=" * 72)
    for strat in ["vector", "fulltext", "hybrid"]:
        print(f"\n########## 策略：{strat} ##########")
        for q in QUESTIONS[:4]:
            hits = engine.search(q["question"], strat, rerank="tfidf")
            print(f"  Q: {q['question'][:30]}...")
            print(f"    Top1: ({hits[0]['score']:.3f}) {hits[0]['text'][:60]}")
    print("\n" + "=" * 72)


if __name__ == "__main__":
    main()
