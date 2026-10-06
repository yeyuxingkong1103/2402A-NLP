# -*- coding: utf-8 -*-
"""
全文检索器 - 倒排索引 + BM25
工单编号: 人工智能 NLP-RAG-混合检索任务

实现:
    1. 轻量级倒排索引 {term → [(chunk_id, tf), ...]}
    2. BM25 打分
    3. 支持多字段检索 (text / title / summary)
    4. 支持布尔查询 AND/OR 和模糊匹配
"""
import os
import json
import math
import logging
from typing import List, Dict, Tuple

import config_v6 as config

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> List[str]:
    try:
        import jieba
        return [t.strip() for t in jieba.cut(text) if t.strip()]
    except ImportError:
        return [c for c in text if c.strip()]


class FullTextRetriever:
    """全文检索器 (倒排索引 + BM25)"""

    def __init__(self):
        self.chunks: List[Dict] = []
        self.inverted_index: Dict[str, List[Tuple[int, int]]] = {}  # term → [(chunk_idx, tf), ...]
        self.doc_count = 0
        self.avg_doc_len = 0.0
        self.k1 = 1.5
        self.b = 0.75

    # ============ 索引构建 ============

    def build_index(self, chunks: List[Dict]):
        self.chunks = chunks
        self.doc_count = len(chunks)

        # 分词 + 构建倒排索引
        doc_lens = []
        for idx, c in enumerate(chunks):
            text = c.get("text", "")
            title = c.get("title", "") or c.get("summary", "")
            # 多字段加权: title 权重更高
            tokens = _tokenize(text) + [t for t in _tokenize(title)] * 2
            doc_lens.append(len(tokens))

            # 词频统计
            tf = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1

            # 倒排索引
            for term, count in tf.items():
                if term not in self.inverted_index:
                    self.inverted_index[term] = []
                self.inverted_index[term].append((idx, count))

        self.avg_doc_len = sum(doc_lens) / max(self.doc_count, 1)
        logger.info(f"[全文检索] 索引构建: {self.doc_count} 块, {len(self.inverted_index)} 词")

    def save_index(self, path: str = None):
        path = path or os.path.join(config.CACHE_DIR, "fulltext_idx.pkl")
        import pickle
        with open(path, "wb") as f:
            pickle.dump({
                "chunks": self.chunks,
                "inverted_index": self.inverted_index,
                "doc_count": self.doc_count,
                "avg_doc_len": self.avg_doc_len,
            }, f)
        logger.info(f"[全文检索] 索引已保存: {path}")

    def load_index(self, path: str = None) -> bool:
        path = path or os.path.join(config.CACHE_DIR, "fulltext_idx.pkl")
        if not os.path.exists(path):
            return False
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.chunks = data["chunks"]
        self.inverted_index = data["inverted_index"]
        self.doc_count = data["doc_count"]
        self.avg_doc_len = data["avg_doc_len"]
        logger.info(f"[全文检索] 索引已加载: {self.doc_count} 块")
        return True

    # ============ BM25 打分 ============

    def _bm25_score(self, query_tokens: List[str]) -> List[float]:
        scores = [0.0] * self.doc_count

        for term in query_tokens:
            if term not in self.inverted_index:
                continue
            postings = self.inverted_index[term]
            df = len(postings)
            idf = math.log((self.doc_count - df + 0.5) / (df + 0.5) + 1)

            for doc_idx, tf in postings:
                doc_len = len(_tokenize(self.chunks[doc_idx].get("text", "")))
                denom = tf + self.k1 * (1 - self.b + self.b * doc_len / max(self.avg_doc_len, 1))
                scores[doc_idx] += idf * (tf * (self.k1 + 1)) / max(denom, 1e-8)

        return scores

    # ============ 检索 ============

    def search(self, query: str, top_k: int = None) -> List[Dict]:
        """全文检索"""
        top_k = top_k or config.TOP_K_FINAL
        query_tokens = _tokenize(query)

        if not query_tokens:
            return []

        scores = self._bm25_score(query_tokens)

        # 取 Top-K
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for idx in top_idx:
            if scores[idx] > 0:
                c = self.chunks[idx]
                results.append({
                    **c,
                    "score": round(float(scores[idx]), 4),
                    "source": "fulltext",
                    "hit_terms": [t for t in query_tokens if t in self.chunks[idx].get("text", "")],
                })
        return results

    def search_boolean(self, must: List[str] = None, should: List[str] = None,
                      top_k: int = None) -> List[Dict]:
        """
        布尔查询: must (AND) + should (OR)

        例: must=["注册资本", "武汉"], should=["万元", "股本"]
        → 必须同时含"注册资本"和"武汉", 最好含"万元"或"股本"
        """
        top_k = top_k or config.TOP_K_FINAL
        must = must or []
        should = should or []

        # 必须同时包含 must 中所有词
        must_docs = None
        for term in must:
            if term in self.inverted_index:
                doc_set = {idx for idx, _ in self.inverted_index[term]}
                must_docs = doc_set if must_docs is None else must_docs & doc_set
            else:
                must_docs = set()  # 有一个词不存在 → 空集
                break

        # 可选包含 should 中任意词 (加分)
        should_docs = set()
        for term in should:
            if term in self.inverted_index:
                should_docs.update(idx for idx, _ in self.inverted_index[term])

        target_docs = must_docs if must_docs is not None else set(range(self.doc_count))
        target_docs = target_docs | should_docs

        # BM25 打分
        scores = self._bm25_score(must + should)

        results = []
        for idx in target_docs:
            c = self.chunks[idx]
            results.append({
                **c,
                "score": round(float(scores[idx]), 4),
                "source": "fulltext_boolean",
            })
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    chunks = [
        {"id": 1, "text": "武汉兴图新科电子股份有限公司注册资本7360万元"},
        {"id": 2, "text": "武汉兴图新科法定代表人XXX"},
        {"id": 3, "text": "本次发行募集资金4亿元用于补充流动资金"},
        {"id": 4, "text": "武汉力源信息技术股份有限公司注册资本8000万元"},
    ]
    ft = FullTextRetriever()
    ft.build_index(chunks)

    print("\n=== '注册资本' ===")
    for r in ft.search("注册资本", top_k=2):
        print(f"  [{r['id']}] score={r['score']:.3f} hit={r.get('hit_terms',[])}")

    print("\n=== 布尔: must=['注册资本','武汉'] should=['万元'] ===")
    for r in ft.search_boolean(must=["注册资本", "武汉"], should=["万元"], top_k=2):
        print(f"  [{r['id']}] score={r['score']:.3f}")
