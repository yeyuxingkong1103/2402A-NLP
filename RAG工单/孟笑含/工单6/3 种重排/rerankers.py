# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-混合检索任务
模块：重排算法集（3 种）
功能：cross_encoder（神经）/ tfidf（关键词）/ adaptive（自适应）
"""

import re
from typing import List, Dict, Any
from collections import Counter
from jieba import cut


class BaseReranker:
    def rerank(self, query, candidates, top_k=3):
        raise NotImplementedError


class CrossEncoderReranker(BaseReranker):
    """重排算法 1：交叉编码器（bge-reranker-base）"""

    def __init__(self, model_name="BAAI/bge-reranker-base"):
        from sentence_transformers import CrossEncoder
        print(f"正在加载 CrossEncoder：{model_name}")
        self.model = CrossEncoder(model_name)
        print("✅ CrossEncoder 加载完成")

    def rerank(self, query, candidates, top_k=3):
        if not candidates:
            return []
        pairs = [[query, c["content"]] for c in candidates]
        scores = self.model.predict(pairs)
        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s)
        return sorted(candidates, key=lambda x: -x["rerank_score"])[:top_k]


class TFIDFReranker(BaseReranker):
    """重排算法 2：TF-IDF 关键词加权"""

    def rerank(self, query, candidates, top_k=3):
        if not candidates:
            return []
        q_tokens = set(cut(query))
        for c in candidates:
            content_tokens = list(cut(c["content"]))
            if not content_tokens:
                c["rerank_score"] = 0.0
                continue
            counter = Counter(content_tokens)
            score = sum(counter.get(t, 0) for t in q_tokens) / len(content_tokens)
            c["rerank_score"] = float(score)
        return sorted(candidates, key=lambda x: -x["rerank_score"])[:top_k]


class AdaptiveReranker(BaseReranker):
    """重排算法 3：自适应（根据问题类型选策略）"""

    def __init__(self, cross_encoder, tfidf):
        self.ce = cross_encoder
        self.tfidf = tfidf

    def rerank(self, query, candidates, top_k=3):
        # 数值型问题 → TF-IDF（关键词精准）
        # 其他问题 → CrossEncoder（语义理解）
        is_numeric = any(kw in query for kw in ["多少", "几", "比例", "占比", "比重", "金额"])
        if is_numeric:
            print("  [自适应] 数值型问题 → 用 TF-IDF")
            return self.tfidf.rerank(query, candidates, top_k)
        else:
            print("  [自适应] 语义型问题 → 用 CrossEncoder")
            return self.ce.rerank(query, candidates, top_k)


def build_reranker(method: str):
    """工厂方法"""
    if method == "cross_encoder":
        return CrossEncoderReranker()
    elif method == "tfidf":
        return TFIDFReranker()
    elif method == "adaptive":
        ce = CrossEncoderReranker()
        tfidf = TFIDFReranker()
        return AdaptiveReranker(ce, tfidf)
    else:
        raise ValueError(f"未知重排算法：{method}")


if __name__ == "__main__":
    test_candidates = [
        {"content": "武汉兴图新科注册资本5,225万元", "page": 56},
        {"content": "公司成立于2004年", "page": 52},
        {"content": "法定代表人程家明", "page": 52},
    ]

    print("=" * 60)
    print("测试 1：CrossEncoder 重排")
    print("=" * 60)
    ce = CrossEncoderReranker()
    for r in ce.rerank("注册资本是多少？", [dict(c) for c in test_candidates]):
        print(f"  第{r['page']}页 score={r['rerank_score']:.4f}")
        print(f"    {r['content'][:60]}")

    print()
    print("=" * 60)
    print("测试 2：TF-IDF 重排")
    print("=" * 60)
    tfidf = TFIDFReranker()
    for r in tfidf.rerank("注册资本是多少？", [dict(c) for c in test_candidates]):
        print(f"  第{r['page']}页 score={r['rerank_score']:.4f}")
        print(f"    {r['content'][:60]}")
