# -*- coding: utf-8 -*-
"""
向量检索模块
工单编号: 人工智能 NLP-RAG-基于 PDF 文档的问答系统

功能:
    1. 将文本块向量化 (TF-IDF)
    2. 持久化索引
    3. 基于 Top-K 余弦相似度检索
"""
import os
import json
import pickle
import logging
from typing import List, Dict

import config

logger = logging.getLogger(__name__)


class VectorRetriever:
    """基于 TF-IDF + 余弦相似度的向量检索器"""

    def __init__(self):
        self.chunks: List[Dict] = []
        self.vectorizer = None
        self.tfidf_matrix = None

    def build_index(self, chunks: List[Dict]):
        """构建 TF-IDF 索引"""
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.chunks = chunks
        texts = [c["text"] for c in chunks]
        # 使用 jieba 分词增强中文检索效果 (如未安装则回退到字符级)
        try:
            import jieba
            tokens = [" ".join(jieba.cut(t)) for t in texts]
        except ImportError:
            logger.warning("未安装 jieba, 使用字符级分词")
            tokens = [" ".join(list(t)) for t in texts]

        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(tokens)
        logger.info(f"TF-IDF 索引构建完成, 形状: {self.tfidf_matrix.shape}")

    def save_index(self, index_path: str = None, chunks_path: str = None):
        """持久化索引"""
        index_path = index_path or config.INDEX_PATH
        chunks_path = chunks_path or config.CHUNKS_PATH
        with open(index_path, "wb") as f:
            pickle.dump({"vectorizer": self.vectorizer,
                          "tfidf_matrix": self.tfidf_matrix}, f)
        with open(chunks_path, "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)
        logger.info(f"索引与 chunks 已持久化")

    def load_index(self, index_path: str = None, chunks_path: str = None):
        """加载持久化索引"""
        index_path = index_path or config.INDEX_PATH
        chunks_path = chunks_path or config.CHUNKS_PATH
        if not (os.path.exists(index_path) and os.path.exists(chunks_path)):
            return False
        with open(index_path, "rb") as f:
            data = pickle.load(f)
            self.vectorizer = data["vectorizer"]
            self.tfidf_matrix = data["tfidf_matrix"]
        with open(chunks_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)
        logger.info(f"已加载索引, 共 {len(self.chunks)} 个文本块")
        return True

    def search(self, query: str, top_k: int = None) -> List[Dict]:
        """
        检索 Top-K 文本块

        Args:
            query: 用户问题
            top_k: 返回数量

        Returns:
            List[{"id", "page", "text", "score"}]
        """
        from sklearn.metrics.pairwise import cosine_similarity

        if self.vectorizer is None:
            raise RuntimeError("索引未构建, 请先调用 build_index 或 load_index")
        top_k = top_k or config.TOP_K

        try:
            import jieba
            query_tokens = " ".join(jieba.cut(query))
        except ImportError:
            query_tokens = " ".join(list(query))

        query_vec = self.vectorizer.transform([query_tokens])
        scores = cosine_similarity(query_vec, self.tfidf_matrix)[0]
        top_indices = scores.argsort()[-top_k:][::-1]
        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            chunk = self.chunks[idx]
            results.append({
                "id": chunk["id"],
                "page": chunk["page"],
                "text": chunk["text"],
                "score": float(scores[idx]),
            })
        return results


def get_retriever() -> VectorRetriever:
    """获取检索器单例 (带索引构建/加载)"""
    retriever = VectorRetriever()
    if retriever.load_index():
        return retriever
    # 索引不存在, 重新解析 PDF 构建
    import pdf_parser
    chunks = pdf_parser.parse_and_save()
    retriever.build_index(chunks)
    retriever.save_index()
    return retriever


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    retriever = get_retriever()
    results = retriever.search("武汉兴图新科电子科技有限公司的注册资本是多少")
    for r in results:
        print(f"[page={r['page']}, score={r['score']:.3f}] {r['text'][:80]}")
