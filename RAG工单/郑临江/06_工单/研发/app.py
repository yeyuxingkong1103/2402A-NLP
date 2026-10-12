# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-混合检索任务
主程序：向量检索（召回+重排）、全文检索、混合检索的配置与应用。
用法：
  python app.py                  # 演示三种检索策略
  python app.py --vector         # 仅向量检索
  python app.py --fulltext       # 仅全文检索
  python app.py --hybrid         # 混合检索
"""
import sys

import config
from pdf_parser import load_pdf_text, chunk_text
from embedder import get_embedder
from vector_retriever import VectorRetriever, TfidfReranker, LLMReranker, AdaptiveFeedbackReranker
from fulltext_retriever import FullTextRetriever
from hybrid import HybridRetriever
from llm import LLM, build_rag_prompt


def build_retrievers():
    docs = []
    for pdf in [config.PDF1, config.PDF2]:
        print(f"[RAG] 解析 PDF：{pdf}")
        docs.extend(chunk_text(load_pdf_text(pdf), config.CHUNK_SIZE, config.CHUNK_OVERLAP))
    print(f"[RAG] 共 {len(docs)} 个文本块")

    embedder = get_embedder(config.EMBED_MODEL)
    vec = VectorRetriever(embedder)
    vec.build(docs)
    ft = FullTextRetriever(docs)

    rerankers = {
        "tfidf": TfidfReranker(docs),
        "llm": LLMReranker(LLM()),
        "adaptive": AdaptiveFeedbackReranker(),
    }
    hybrid = HybridRetriever(vec, ft, config.VECTOR_WEIGHT, config.FULLTEXT_WEIGHT)
    return docs, vec, ft, rerankers, hybrid


def show(label, results):
    print(f"\n--- {label} ---")
    for doc, score in results:
        print(f"  [{score:.3f}] {doc[:80].replace(chr(10), ' ')}")


def demo():
    docs, vec, ft, rerankers, hybrid = build_retrievers()
    llm = LLM()
    for q in config.QUESTIONS:
        print(f"\n==================== 问题(id={q['id']}) ====================")
        print("Q:", q["question"])
        # 向量召回 + 重排
        recalled = vec.recall(q["question"], top_k=config.RECALL_TOP_K)
        reranked = rerankers["tfidf"].rerank(q["question"], [d for d, _ in recalled], top_k=config.RERANK_TOP_K)
        show("向量检索(召回+TF-IDF重排)", reranked)
        # 全文检索
        show("全文检索", ft.search(q["question"], top_k=3))
        # 混合检索
        show("混合检索(加权融合)", hybrid.search(q["question"]))
        # 生成答案
        hits = hybrid.search(q["question"])
        answer = llm.generate(build_rag_prompt(q["question"], hits))
        print(f"答案：{answer[:150].replace(chr(10), ' ')}")


if __name__ == "__main__":
    demo()
