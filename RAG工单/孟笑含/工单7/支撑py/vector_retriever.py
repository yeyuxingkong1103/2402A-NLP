# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化
模块：向量检索模块（优化版）
功能：bge-m3 向量检索 + BM25 关键词检索 + RRF 融合
"""

import numpy as np
from typing import List, Dict, Any
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from jieba import cut


class VectorRetriever:
    """向量检索器（混合检索优化版）"""

    def __init__(self, model_name: str = "BAAI/bge-m3"):
        print(f"正在加载嵌入模型：{model_name}")
        self.encoder = SentenceTransformer(
            model_name,
            model_kwargs={"use_safetensors": True}
        )
        try:
            self.dimension = self.encoder.get_embedding_dimension()
        except AttributeError:
            self.dimension = self.encoder.get_sentence_embedding_dimension()
        self.index = None
        self.chunks = []
        self.bm25 = None
        print(f"✅ 模型加载完成，向量维度：{self.dimension}")

    def build_index(self, chunks: List[Dict[str, Any]]):
        """构建向量索引 + BM25 索引"""
        self.chunks = chunks
        texts = [c["content"] for c in chunks]

        print(f"正在向量化 {len(texts)} 个文本块...")
        embeddings = self.encoder.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=32
        )
        embeddings = np.array(embeddings).astype("float32")

        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)
        print(f"✅ FAISS 索引构建完成：{len(chunks)} 块，维度 {self.dimension}")

        print(f"正在构建 BM25 索引...")
        tokenized_corpus = [list(cut(t)) for t in texts]
        self.bm25 = BM25Okapi(tokenized_corpus)
        print(f"✅ BM25 索引构建完成")

    def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """混合检索：向量 Top-K + BM25 Top-K，RRF 融合"""
        if self.index is None:
            raise ValueError("请先调用 build_index()")

        candidate_k = max(top_k * 2, 20)

        # 1) 向量检索
        qvec = self.encoder.encode([query], normalize_embeddings=True)
        qvec = np.array(qvec).astype("float32")
        v, idx = self.index.search(qvec, candidate_k)
        vec_results = {int(i): float(s) for s, i in zip(v[0], idx[0]) if i >= 0}

        # 2) BM25 检索
        tokens = list(cut(query))
        bm25_scores = self.bm25.get_scores(tokens)
        bm25_idx = np.argsort(bm25_scores)[::-1][:candidate_k]
        bm25_results = {int(i): float(bm25_scores[i]) for i in bm25_idx}

        # 3) RRF 融合（k=60）
        RRF_K = 60
        rrf = {}
        for rank, i in enumerate(sorted(vec_results.keys(), key=lambda x: -vec_results[x])):
            rrf[i] = rrf.get(i, 0) + 1 / (RRF_K + rank + 1)
        for rank, i in enumerate(bm25_idx.tolist()):
            if bm25_scores[i] > 0:
                rrf[i] = rrf.get(i, 0) + 1 / (RRF_K + rank + 1)

        sorted_ids = sorted(rrf.keys(), key=lambda x: -rrf[x])[:top_k]

        results = []
        for i in sorted_ids:
            item = dict(self.chunks[i])
            item["score"] = float(rrf[i])
            item["vec_score"] = vec_results.get(i, 0.0)
            item["bm25_score"] = bm25_results.get(i, 0.0)
            results.append(item)

        return results


if __name__ == "__main__":
    from pdf_parser import PDFParser
    parser = PDFParser("./data/招股说明书1.pdf")
    pages = parser.extract_text()
    chunks = parser.chunk_text(pages, chunk_size=300, overlap=80)
    retriever = VectorRetriever()
    retriever.build_index(chunks)
    res = retriever.retrieve("注册资本是多少？", top_k=5)
    for i, r in enumerate(res, 1):
        print(f"[{i}] 第{r['page']}页 RRF={r['score']:.4f} VEC={r['vec_score']:.4f} BM25={r['bm25_score']:.4f}")
        print(f"    {r['content'][:80]}...")
