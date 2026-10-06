# -*- coding: utf-8 -*-
"""
混合检索器 - 向量 + 全文 + 融合
工单编号: 人工智能 NLP-RAG-混合检索任务

融合方法:
    1. weighted: 加权平均 score = α*vector + (1-α)*fulltext
    2. rrf: Reciprocal Rank Fusion
    3. vote: 投票机制 (两路都命中加分)
"""
import logging
from typing import List, Dict

import config_v6 as config
import vector_retriever_v6
import fulltext_retriever

logger = logging.getLogger(__name__)


class HybridRetriever:
    """混合检索器"""

    def __init__(self, vector_weight: float = None, fulltext_weight: float = None,
                 fusion_method: str = None):
        self.vector_weight = vector_weight if vector_weight is not None else config.VECTOR_WEIGHT
        self.fulltext_weight = fulltext_weight if fulltext_weight is not None else config.FULLTEXT_WEIGHT
        self.fusion_method = fusion_method or config.FUSION_METHOD

        self.vector_retriever = vector_retriever_v6.VectorRetriever()
        self.fulltext_retriever = fulltext_retriever.FullTextRetriever()

    def build_index(self, chunks: List[Dict]):
        self.vector_retriever.build_index(chunks)
        self.fulltext_retriever.build_index(chunks)
        logger.info(f"[混合检索] 索引构建完成: {len(chunks)} 块, 融合={self.fusion_method}")

    def save_index(self):
        self.vector_retriever.save_index()
        self.fulltext_retriever.save_index()

    def load_index(self) -> bool:
        v_ok = self.vector_retriever.load_index()
        f_ok = self.fulltext_retriever.load_index()
        return v_ok and f_ok

    def _normalize_scores(self, results: List[Dict]) -> List[Dict]:
        """分数归一化到 [0, 1]"""
        if not results:
            return results
        max_s = max(r.get("score", 0) for r in results)
        min_s = min(r.get("score", 0) for r in results)
        for r in results:
            if max_s > min_s:
                r["norm_score"] = (r.get("score", 0) - min_s) / (max_s - min_s)
            else:
                r["norm_score"] = 1.0
        return results

    def _fuse_weighted(self, v_results: List[Dict], f_results: List[Dict]) -> List[Dict]:
        """加权融合"""
        v_results = self._normalize_scores(v_results)
        f_results = self._normalize_scores(f_results)

        # 按 id 合并
        merged = {}
        for r in v_results:
            cid = str(r.get("id", ""))
            merged[cid] = {**r, "v_score": r.get("norm_score", 0), "f_score": 0}
        for r in f_results:
            cid = str(r.get("id", ""))
            if cid in merged:
                merged[cid]["f_score"] = r.get("norm_score", 0)
            else:
                merged[cid] = {**r, "v_score": 0, "f_score": r.get("norm_score", 0)}

        # 计算融合分数
        for cid, item in merged.items():
            item["fused_score"] = (
                self.vector_weight * item["v_score"] +
                self.fulltext_weight * item["f_score"]
            )
            # 双路命中额外加分
            if item["v_score"] > 0 and item["f_score"] > 0:
                item["fused_score"] *= 1.2  # +20%

        results = sorted(merged.values(), key=lambda x: x["fused_score"], reverse=True)
        return results

    def _fuse_rrf(self, v_results: List[Dict], f_results: List[Dict],
                  k: int = 60) -> List[Dict]:
        """Reciprocal Rank Fusion"""
        def get_rrf(rank: int) -> float:
            return 1.0 / (k + rank)

        merged = {}
        for rank, r in enumerate(v_results):
            cid = str(r.get("id", ""))
            rrf_v = get_rrf(rank + 1)
            if cid in merged:
                merged[cid]["fused_score"] += rrf_v
                merged[cid]["v_rank"] = rank + 1
            else:
                merged[cid] = {**r, "fused_score": rrf_v, "v_rank": rank + 1, "f_rank": None}

        for rank, r in enumerate(f_results):
            cid = str(r.get("id", ""))
            rrf_f = get_rrf(rank + 1)
            if cid in merged:
                merged[cid]["fused_score"] += rrf_f
                merged[cid]["f_rank"] = rank + 1
            else:
                merged[cid] = {**r, "fused_score": rrf_f, "v_rank": None, "f_rank": rank + 1}

        results = sorted(merged.values(), key=lambda x: x["fused_score"], reverse=True)
        return results

    def _fuse_vote(self, v_results: List[Dict], f_results: List[Dict]) -> List[Dict]:
        """投票机制: 两路都命中 → 高票"""
        merged = {}
        for r in v_results:
            cid = str(r.get("id", ""))
            merged[cid] = {**r, "votes": 1, "v_score": r.get("score", 0), "f_score": 0}
        for r in f_results:
            cid = str(r.get("id", ""))
            if cid in merged:
                merged[cid]["votes"] += 1
                merged[cid]["f_score"] = r.get("score", 0)
            else:
                merged[cid] = {**r, "votes": 1, "v_score": 0, "f_score": r.get("score", 0)}

        # 投票 → 融合分数
        for cid, item in merged.items():
            norm_v = item["v_score"]  # 原始分
            norm_f = item["f_score"]
            item["fused_score"] = item["votes"] * 0.5 + 0.3 * norm_v + 0.2 * norm_f
            if item["votes"] == 2:
                item["fused_score"] += 0.3  # 双路命中额外加分

        results = sorted(merged.values(), key=lambda x: x["fused_score"], reverse=True)
        return results

    def search(self, query: str, top_k: int = None) -> List[Dict]:
        """混合检索完整流程"""
        top_k = top_k or config.TOP_K_FINAL

        # 1. 两路并行检索
        logger.info(f"[混合检索] query='{query[:30]}' strategy={config.RETRIEVAL_STRATEGY}")
        v_results = self.vector_retriever.recall(query, top_k=config.TOP_K_RECALL)
        f_results = self.fulltext_retriever.search(query, top_k=config.TOP_K_RECALL)
        logger.info(f"  向量召回 {len(v_results)}, 全文召回 {len(f_results)}")

        # 2. 融合
        if self.fusion_method == "weighted":
            fused = self._fuse_weighted(v_results, f_results)
        elif self.fusion_method == "rrf":
            fused = self._fuse_rrf(v_results, f_results)
        elif self.fusion_method == "vote":
            fused = self._fuse_vote(v_results, f_results)
        else:
            fused = self._fuse_weighted(v_results, f_results)

        # 3. 取 Top-K
        final = fused[:top_k]
        logger.info(f"[混合检索] 融合完成, 返回 {len(final)} 条")
        return final

    # ============ 策略选择 ============

    def search_by_strategy(self, query: str, strategy: str = None,
                           top_k: int = None) -> List[Dict]:
        """
        按指定策略检索

        Args:
            strategy: vector / fulltext / hybrid (None → 用 config)
        """
        strategy = strategy or config.RETRIEVAL_STRATEGY
        top_k = top_k or config.TOP_K_FINAL

        if strategy == "vector":
            return self.vector_retriever.search(query, top_k)
        elif strategy == "fulltext":
            return self.fulltext_retriever.search(query, top_k)
        else:  # hybrid
            return self.search(query, top_k)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    chunks = [
        {"id": 1, "text": "武汉兴图新科电子股份有限公司注册资本7360万元"},
        {"id": 2, "text": "武汉兴图新科法定代表人XXX"},
        {"id": 3, "text": "本次发行募集资金4亿元补充流动资金"},
        {"id": 4, "text": "武汉力源信息技术股份有限公司注册资本8000万元"},
    ]

    for method in ["weighted", "rrf", "vote"]:
        print(f"\n{'='*50}")
        print(f"融合方法: {method}")
        hr = HybridRetriever(fusion_method=method)
        hr.build_index(chunks)
        results = hr.search("注册资本 武汉", top_k=3)
        for r in results:
            print(f"  [{r['id']}] fused={r.get('fused_score',0):.3f} | {r['text'][:40]}")
