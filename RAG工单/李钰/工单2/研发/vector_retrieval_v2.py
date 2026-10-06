# -*- coding: utf-8 -*-
"""
混合检索模块 V2 - BM25 + TF-IDF + Rerank
工单编号: 人工智能 NLP-RAG-基于 PDF 文档的问答系统优化

核心改进 (对比 V1):
    1. BM25 关键词检索: 精确关键词匹配效果好
    2. TF-IDF 向量检索: 语义相似性补充
    3. 混合融合: 两路分数加权合并
    4. Rerank: 用 query 关键词覆盖率重排序 Top-20 → Top-5
"""
import os
import json
import pickle
import math
import logging
from typing import List, Dict, Tuple

import config_v2 as config

logger = logging.getLogger(__name__)

# 中文分词 (优先 jieba)
def _tokenize(text: str) -> List[str]:
    try:
        import jieba
        return [t.strip() for t in jieba.cut(text) if t.strip()]
    except ImportError:
        return [c for c in text if c.strip()]


class BM25Retriever:
    """轻量级 BM25 实现 (Okapi Best Matching)"""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_count = 0
        self.doc_len_avg = 0.0
        self.doc_tokens: List[List[str]] = []
        self.doc_tf: List[Dict[str, int]] = []   # 每个文档的词频
        self.df: Dict[str, int] = {}              # 每个词的文档频率
        self.idf: Dict[str, float] = {}

    def fit(self, tokenized_docs: List[List[str]]):
        self.doc_tokens = tokenized_docs
        self.doc_count = len(tokenized_docs)
        self.doc_len_avg = sum(len(d) for d in tokenized_docs) / max(self.doc_count, 1)

        # 计算 tf 和 df
        for tokens in tokenized_docs:
            tf = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            self.doc_tf.append(tf)
            for term in tf:
                self.df[term] = self.df.get(term, 0) + 1

        # 计算 IDF
        for term, df in self.df.items():
            self.idf[term] = math.log(
                (self.doc_count - df + 0.5) / (df + 0.5) + 1
            )
        logger.info(f"BM25 索引完成: {self.doc_count} 文档, {len(self.idf)} 词")

    def score(self, query_tokens: List[str]) -> List[float]:
        """对所有文档计算 BM25 分数"""
        scores = []
        for i, tokens in enumerate(self.doc_tokens):
            doc_len = len(tokens)
            tf = self.doc_tf[i]
            score = 0.0
            for qt in query_tokens:
                if qt not in tf:
                    continue
                idf = self.idf.get(qt, 0.0)
                f = tf[qt]
                denom = f + self.k1 * (1 - self.b + self.b * doc_len / max(self.doc_len_avg, 1))
                score += idf * (f * (self.k1 + 1)) / max(denom, 1e-8)
            scores.append(score)
        return scores


class HybridRetriever:
    """混合检索器: BM25 + TF-IDF + Rerank"""

    def __init__(self):
        self.chunks: List[Dict] = []
        self.bm25 = BM25Retriever()
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None

    def build_index(self, chunks: List[Dict]):
        """构建混合索引"""
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.chunks = chunks
        texts = [c["text"] for c in chunks]
        tokenized = [_tokenize(t) for t in texts]

        # BM25
        self.bm25.fit(tokenized)

        # TF-IDF (空格拼接后的 token 序列)
        joined = [" ".join(t) for t in tokenized]
        self.tfidf_vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(joined)
        logger.info(f"[V2 混合索引] 完成, {len(chunks)} 个块")

    def save_index(self, path: str = None):
        path = path or config.INDEX_PATH
        with open(path, "wb") as f:
            pickle.dump({
                "bm25": self.bm25,
                "tfidf_vectorizer": self.tfidf_vectorizer,
                "tfidf_matrix": self.tfidf_matrix,
            }, f)
        with open(os.path.join(os.path.dirname(path), "chunks_v2.json"), "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)
        logger.info(f"[V2 索引] 已持久化")

    def load_index(self, path: str = None) -> bool:
        path = path or config.INDEX_PATH
        chunks_path = os.path.join(os.path.dirname(path), "chunks_v2.json")
        if not (os.path.exists(path) and os.path.exists(chunks_path)):
            return False
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.bm25 = data["bm25"]
            self.tfidf_vectorizer = data["tfidf_vectorizer"]
            self.tfidf_matrix = data["tfidf_matrix"]
        with open(chunks_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)
        logger.info(f"[V2 索引] 已加载, {len(self.chunks)} 个块")
        return True

    def _normalize(self, scores: List[float]) -> List[float]:
        """分数归一化到 [0, 1]"""
        if not scores:
            return []
        max_s = max(scores)
        if max_s <= 0:
            return [0.0] * len(scores)
        return [s / max_s for s in scores]

    def search(self, query: str, top_k: int = None, raw_k: int = None) -> List[Dict]:
        """
        混合检索 + Rerank

        Args:
            query: 用户问题
            top_k: 最终返回数量
            raw_k: 各路检索候选数

        Returns:
            List[{"id", "page", "text", "score", "rank_score"}]
        """
        from sklearn.metrics.pairwise import cosine_similarity

        top_k = top_k or config.TOP_K_FINAL
        raw_k = raw_k or config.TOP_K_RAW

        query_tokens = _tokenize(query)

        # 1. BM25 检索
        bm25_raw = self.bm25.score(query_tokens)
        bm25_norm = self._normalize(bm25_raw)
        bm25_indices = sorted(range(len(bm25_norm)),
                              key=lambda i: bm25_norm[i], reverse=True)[:raw_k]

        # 2. TF-IDF 检索
        query_str = " ".join(query_tokens)
        query_vec = self.tfidf_vectorizer.transform([query_str])
        tfidf_scores = cosine_similarity(query_vec, self.tfidf_matrix)[0].tolist()
        tfidf_norm = self._normalize(tfidf_scores)
        tfidf_indices = sorted(range(len(tfidf_norm)),
                               key=lambda i: tfidf_norm[i], reverse=True)[:raw_k]

        # 3. 融合去重
        merged_indices = set(bm25_indices + tfidf_indices)
        candidates = []
        for idx in merged_indices:
            # 同时出现在两路则加分
            in_bm25 = idx in bm25_indices
            in_tfidf = idx in tfidf_indices
            bm25_s = bm25_norm[idx] if in_bm25 else 0.0
            tfidf_s = tfidf_norm[idx] if in_tfidf else 0.0
            hybrid = config.BM25_WEIGHT * bm25_s + config.TFIDF_WEIGHT * tfidf_s
            # 双路命中额外加分
            if in_bm25 and in_tfidf:
                hybrid *= 1.2
            candidates.append((idx, hybrid))

        # 4. Rerank: query 关键词覆盖率调整
        candidates = self._rerank(candidates, query_tokens)

        # 5. 取 Top-K
        candidates.sort(key=lambda x: x[1], reverse=True)
        top = candidates[:top_k]

        results = []
        for idx, score in top:
            chunk = self.chunks[idx]
            results.append({
                "id": chunk["id"],
                "page": chunk["page"],
                "text": chunk["text"],
                "score": round(float(score), 4),
                "type": chunk.get("type", "text"),
            })
        return results

    def _rerank(self, candidates: List[Tuple[int, float]],
                query_tokens: List[str]) -> List[Tuple[int, float]]:
        """
        简单 Rerank: query 关键词覆盖率越高, 分数提升越多
        """
        if not query_tokens:
            return candidates
        reranked = []
        for idx, base_score in candidates:
            chunk_text = self.chunks[idx]["text"]
            # 计算 query 关键词在文档中的出现率
            hit = sum(1 for t in query_tokens if t in chunk_text)
            coverage = hit / len(query_tokens)
            # 覆盖率权重, 最高 +50%
            boost = 1.0 + coverage * 0.5
            reranked.append((idx, base_score * boost))
        return reranked


def get_hybrid_retriever() -> HybridRetriever:
    """获取混合检索器单例"""
    retriever = HybridRetriever()
    if retriever.load_index():
        return retriever
    # 索引不存在, 重新解析构建
    import pdf_parser_v2
    chunks = pdf_parser_v2.parse_and_save_v2()
    retriever.build_index(chunks)
    retriever.save_index()
    return retriever


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    retriever = get_hybrid_retriever()
    q = "武汉兴图新科电子股份有限公司的注册资本是多少"
    results = retriever.search(q, top_k=5)
    for r in results:
        print(f"[p{r['page']}, type={r['type']}, score={r['score']:.3f}] {r['text'][:80]}")
