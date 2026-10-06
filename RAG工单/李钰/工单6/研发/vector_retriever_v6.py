# -*- coding: utf-8 -*-
"""
向量检索器 - 召回 + 重排
工单编号: 人工智能 NLP-RAG-混合检索任务

支持: TF-IDF (零依赖) / sentence-transformers (可选)
"""
import os
import json
import pickle
import logging
import numpy as np
from typing import List, Dict

import config_v6 as config
import rerankers

logger = logging.getLogger(__name__)


class VectorRetriever:
    """向量检索器 (召回 + 重排)"""

    def __init__(self, embedding_model: str = None, reranker_name: str = None):
        self.embedding_model = embedding_model or config.EMBEDDING_MODEL
        self.reranker_name = reranker_name or config.RERANKER
        self.chunks: List[Dict] = []
        self._vectorizer = None
        self._matrix = None
        self._sentence_transformer = None
        self._reranker = rerankers.get_reranker(self.reranker_name)

    # ============ 嵌入 ============

    def _init_embedding(self):
        if self.embedding_model == "tfidf":
            from sklearn.feature_extraction.text import TfidfVectorizer
            try:
                import jieba
                def tokenizer(text):
                    return [t for t in jieba.cut(text) if t.strip()]
            except ImportError:
                def tokenizer(text):
                    return list(text)
            self._vectorizer = TfidfVectorizer(tokenizer=tokenizer, lowercase=False)
        elif self.embedding_model in ("bge", "m3e", "sentence-transformers"):
            try:
                from sentence_transformers import SentenceTransformer
                model_map = {
                    "bge": "BAAI/bge-small-zh",
                    "m3e": "moka-ai/m3e-small",
                }
                model_name = model_map.get(self.embedding_model, "BAAI/bge-small-zh")
                logger.info(f"加载嵌入模型: {model_name}")
                self._sentence_transformer = SentenceTransformer(model_name)
            except ImportError:
                logger.warning("sentence-transformers 未安装, 降级为 TF-IDF")
                self.embedding_model = "tfidf"
                self._init_embedding()

    def _encode_texts(self, texts: List[str]) -> np.ndarray:
        if self.embedding_model == "tfidf":
            return self._vectorizer.transform(texts).toarray()
        elif self._sentence_transformer:
            return self._sentence_transformer.encode(texts, normalize_embeddings=True)
        return np.array([])

    # ============ 索引构建 ============

    def build_index(self, chunks: List[Dict]):
        self.chunks = chunks
        self._init_embedding()
        texts = [c.get("text", "") for c in chunks]
        self._matrix = self._encode_texts(texts)
        logger.info(f"[向量检索] 索引构建完成: {len(chunks)} 块, 模型={self.embedding_model}")

    def save_index(self, path: str = None):
        path = path or os.path.join(config.CACHE_DIR, f"vector_idx_{self.embedding_model}.pkl")
        data = {
            "chunks": self.chunks,
            "embedding_model": self.embedding_model,
            "matrix": self._matrix,
        }
        if self.embedding_model == "tfidf":
            data["vectorizer"] = self._vectorizer
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"[向量检索] 索引已保存: {path}")

    def load_index(self, path: str = None) -> bool:
        path = path or os.path.join(config.CACHE_DIR, f"vector_idx_{self.embedding_model}.pkl")
        if not os.path.exists(path):
            return False
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.chunks = data["chunks"]
        self.embedding_model = data["embedding_model"]
        self._matrix = data["matrix"]
        if self.embedding_model == "tfidf":
            self._vectorizer = data["vectorizer"]
        else:
            self._init_embedding()
        logger.info(f"[向量检索] 索引已加载: {len(self.chunks)} 块")
        return True

    # ============ 召回 + 重排 ============

    def _cosine_similarity(self, query_vec: np.ndarray) -> np.ndarray:
        if self.embedding_model == "tfidf":
            from sklearn.metrics.pairwise import cosine_similarity
            return cosine_similarity(query_vec, self._matrix)[0]
        else:
            # 已归一化, 直接点积
            return np.dot(self._matrix, query_vec.T).flatten()

    def recall(self, query: str, top_k: int = None) -> List[Dict]:
        """高效召回"""
        top_k = top_k or config.TOP_K_RECALL
        query_vec = self._encode_texts([query])
        scores = self._cosine_similarity(query_vec)

        # 取 Top-K
        top_idx = scores.argsort()[-top_k:][::-1]
        results = []
        for idx in top_idx:
            c = self.chunks[idx]
            results.append({
                **c,
                "score": round(float(scores[idx]), 4),
                "source": "vector",
            })
        return results

    def search(self, query: str, top_k: int = None) -> List[Dict]:
        """召回 + 重排 完整流程"""
        top_k = top_k or config.TOP_K_FINAL

        # 1. 召回 Top-20
        candidates = self.recall(query, top_k=config.TOP_K_RECALL)

        # 2. 重排
        try:
            candidates = self._reranker.rerank(query, candidates)
        except Exception as e:
            logger.warning(f"重排失败: {e}")
            for c in candidates:
                c["rerank_score"] = c.get("score", 0)

        # 3. 取 Top-5
        final = sorted(candidates,
                      key=lambda x: x.get("rerank_score", x.get("score", 0)),
                      reverse=True)[:top_k]
        return final

    def get_reranker_info(self) -> Dict:
        return {
            "embedding_model": self.embedding_model,
            "reranker": self.reranker_name,
            "chunk_count": len(self.chunks),
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # 模拟
    chunks = [
        {"id": 1, "text": "武汉兴图新科电子股份有限公司注册资本7360万元"},
        {"id": 2, "text": "武汉兴图新科法定代表人XXX"},
        {"id": 3, "text": "本次发行募集资金4亿元"},
    ]
    vr = VectorRetriever(embedding_model="tfidf", reranker_name="tfidf")
    vr.build_index(chunks)
    results = vr.search("注册资本是多少", top_k=2)
    for r in results:
        print(f"  [{r['id']}] score={r.get('score',0):.3f} "
              f"rerank={r.get('rerank_score',0):.3f} | {r['text'][:40]}")
