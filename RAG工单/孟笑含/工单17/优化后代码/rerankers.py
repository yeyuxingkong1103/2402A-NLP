# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-混合检索任务
模块：重排算法集（3 种）
功能：cross_encoder（神经）/ tfidf（关键词）/ adaptive（自适应）
"""

import re
import threading
from typing import List, Dict, Any
from collections import Counter
from jieba import cut


class BaseReranker:
    def rerank(self, query, candidates, top_k=3):
        raise NotImplementedError


class CrossEncoderReranker(BaseReranker):
    """重排算法 1：交叉编码器（bge-reranker-base）

    工单17优化：
    - _model 类属性单例，避免并发时重复加载模型
    - _model_lock 线程锁，保证加载线程安全
    - rerank 后显式 torch.cuda.empty_cache()，释放 GPU 缓存
    """

    _model = None
    _model_name = ""
    _model_lock = threading.Lock()

    def __init__(self, model_name="BAAI/bge-reranker-base"):
        if not CrossEncoderReranker._model or model_name != CrossEncoderReranker._model_name:
            with CrossEncoderReranker._model_lock:
                if not CrossEncoderReranker._model or model_name != CrossEncoderReranker._model_name:
                    from sentence_transformers import CrossEncoder
                    print(f"正在加载 CrossEncoder：{model_name}")
                    CrossEncoderReranker._model = CrossEncoder(model_name)
                    CrossEncoderReranker._model_name = model_name
                    print("✅ CrossEncoder 加载完成")
        self.model = CrossEncoderReranker._model

    def rerank(self, query, candidates, top_k=3):
        if not candidates:
            return []
        pairs = [[query, c["content"]] for c in candidates]
        scores = self.model.predict(pairs)
        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s)
        result = sorted(candidates, key=lambda x: -x["rerank_score"])[:top_k]
        self.torch_empty_cache()
        return result

    def torch_empty_cache(self):
        """工单17优化：显式释放 GPU 缓存，避免显存持续占用"""
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"Error emptying cache: {e}")


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


# 工单17优化：进程内 reranker 单例缓存
_reranker_cache = {}
_reranker_cache_lock = threading.Lock()


def build_reranker(method: str):
    """工厂方法（工单17优化：进程内单例缓存）"""
    if method in _reranker_cache:
        return _reranker_cache[method]
    with _reranker_cache_lock:
        if method in _reranker_cache:
            return _reranker_cache[method]
        if method == "cross_encoder":
            r = CrossEncoderReranker()
        elif method == "tfidf":
            r = TFIDFReranker()
        elif method == "adaptive":
            ce = CrossEncoderReranker()
            tfidf = TFIDFReranker()
            r = AdaptiveReranker(ce, tfidf)
        else:
            raise ValueError(f"未知重排算法：{method}")
        _reranker_cache[method] = r
        return r


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
