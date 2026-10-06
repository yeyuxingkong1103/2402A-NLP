# -*- coding: utf-8 -*-
"""
评估框架 - RetrievalEvaluator + QAEvaluator + PerformanceEvaluator
工单编号: 人工智能 NLP-RAG-功能测试及评估
"""
import math
import time
import json
import logging
from typing import List, Dict, Optional, Set

logger = logging.getLogger(__name__)


class RetrievalEvaluator:
    """检索评估: Precision / Recall / MRR / NDCG / HitRate"""

    def __init__(self, top_k: int = 5):
        self.top_k = top_k

    def evaluate(self, retrieved: List[Dict], relevant_ids: Set[str]) -> Dict:
        """
        评估单次检索

        Args:
            retrieved: [{id, text, score, ...}] 检索返回列表
            relevant_ids: 相关 chunk 的 id 集合 (ground truth)

        Returns:
            各指标分数
        """
        k = self.top_k
        top_k_ids = [str(r.get("id", "")) for r in retrieved[:k]]

        # 1. Precision@K
        hits_at_k = sum(1 for cid in top_k_ids if cid in relevant_ids)
        precision_at_k = hits_at_k / max(k, 1)

        # 2. Recall@K
        recall_at_k = hits_at_k / max(len(relevant_ids), 1)

        # 3. HitRate@K (至少命中一个)
        hit_rate = 1.0 if hits_at_k > 0 else 0.0

        # 4. MRR (Mean Reciprocal Rank)
        mrr = 0.0
        for rank, cid in enumerate(top_k_ids, start=1):
            if cid in relevant_ids:
                mrr = 1.0 / rank
                break  # 只取第一个相关块

        # 5. NDCG@K
        ndcg = self._ndcg_at_k(top_k_ids, relevant_ids, k)

        return {
            "precision@k": round(precision_at_k, 4),
            "recall@k": round(recall_at_k, 4),
            "hit_rate@k": round(hit_rate, 4),
            "mrr": round(mrr, 4),
            "ndcg@k": round(ndcg, 4),
            "hits_at_k": hits_at_k,
            "total_relevant": len(relevant_ids),
        }

    def _ndcg_at_k(self, ranked_ids: List[str], relevant_ids: Set[str], k: int) -> float:
        """NDCG@K"""
        # DCG
        dcg = 0.0
        for i, cid in enumerate(ranked_ids[:k], start=1):
            rel = 1.0 if cid in relevant_ids else 0.0
            dcg += rel / math.log2(i + 1)

        # IDCG (理想排序)
        ideal = min(len(relevant_ids), k)
        idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal + 1))

        if idcg == 0:
            return 0.0
        return dcg / idcg

    def evaluate_batch(self, results: List[Dict]) -> Dict:
        """批量评估"""
        all_metrics = {"precision@k": [], "recall@k": [], "hit_rate@k": [],
                       "mrr": [], "ndcg@k": []}

        for r in results:
            retrieved = r.get("retrieved", [])
            relevant = set(r.get("relevant_ids", []))
            m = self.evaluate(retrieved, relevant)
            for key in all_metrics:
                all_metrics[key].append(m[key])

        avg = {k: round(sum(v) / max(len(v), 1), 4) for k, v in all_metrics.items()}
        avg["count"] = len(results)
        return avg


class QAEvaluator:
    """问答评估: 关键词覆盖率 / BLEU / ROUGE-L / 幻觉检测"""

    def __init__(self):
        pass

    def evaluate_keyword_coverage(self, answer: str, ref_keywords: List[str]) -> float:
        """关键词覆盖率"""
        if not answer or not ref_keywords:
            return 0.0
        hit = sum(1 for k in ref_keywords if k in answer)
        return round(hit / len(ref_keywords), 4)

    def evaluate_bleu(self, answer: str, reference: str, n_gram: int = 1) -> float:
        """BLEU 分数 (简化版, 支持 1-gram)"""
        if not answer or not reference:
            return 0.0

        def get_ngrams(text, n):
            chars = [c for c in text if c.strip()]
            return [tuple(chars[i:i+n]) for i in range(len(chars) - n + 1)]

        ans_ngrams = get_ngrams(answer, n_gram)
        ref_ngrams = get_ngrams(reference, n_gram)

        if not ans_ngrams or not ref_ngrams:
            return 0.0

        # 计算 precision
        ref_counts = {}
        for ng in ref_ngrams:
            ref_counts[ng] = ref_counts.get(ng, 0) + 1

        clipped = 0
        for ng in ans_ngrams:
            if ng in ref_counts and ref_counts[ng] > 0:
                clipped += 1
                ref_counts[ng] -= 1

        bleu = clipped / max(len(ans_ngrams), 1)
        return round(bleu, 4)

    def evaluate_rouge_l(self, answer: str, reference: str) -> float:
        """ROUGE-L (最长公共子序列覆盖率)"""
        if not answer or not reference:
            return 0.0

        ans_chars = [c for c in answer if c.strip()]
        ref_chars = [c for c in reference if c.strip()]

        # LCS 计算
        m, n = len(ref_chars), len(ans_chars)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if ref_chars[i-1] == ans_chars[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])

        lcs_len = dp[m][n]
        if m == 0:
            return 0.0

        recall = lcs_len / m
        precision = lcs_len / max(n, 1)

        # F1
        if recall + precision == 0:
            return 0.0
        f1 = 2 * recall * precision / (recall + precision)
        return round(f1, 4)

    def evaluate_hallucination(self, answer: str, context_chunks: List[str]) -> float:
        """
        幻觉检测: 答案中有多少内容不在检索上下文中
        返回值越大 → 幻觉越严重
        """
        if not answer or not context_chunks:
            return 1.0  # 无上下文 → 全部算幻觉

        # 简化方法: 将答案分句, 看每个分句是否在上下文中有对应
        answer_sentences = [s.strip() for s in answer.replace("。", "。\n").split("\n") if s.strip()]
        context_text = " ".join(context_chunks)

        hallucinated = 0
        for sent in answer_sentences:
            # 简化: 检查分句的前几个字符是否在上下文中
            key_chars = sent[:10]
            if key_chars not in context_text:
                hallucinated += 1

        ratio = hallucinated / max(len(answer_sentences), 1)
        return round(ratio, 4)

    def evaluate(self, answer: str, reference: str, ref_keywords: List[str],
                 context_chunks: List[str] = None) -> Dict:
        """综合评估"""
        return {
            "keyword_coverage": self.evaluate_keyword_coverage(answer, ref_keywords),
            "bleu_1": self.evaluate_bleu(answer, reference, n_gram=1),
            "rouge_l": self.evaluate_rouge_l(answer, reference),
            "hallucination_ratio": self.evaluate_hallucination(
                answer, context_chunks or []),
        }

    def evaluate_batch(self, results: List[Dict]) -> Dict:
        """批量评估"""
        all_metrics = {"keyword_coverage": [], "bleu_1": [], "rouge_l": [], "hallucination_ratio": []}
        for r in results:
            m = self.evaluate(
                r.get("answer", ""),
                r.get("reference", ""),
                r.get("ref_keywords", []),
                r.get("context_chunks", []),
            )
            for key in all_metrics:
                all_metrics[key].append(m[key])

        avg = {k: round(sum(v) / max(len(v), 1), 4) for k, v in all_metrics.items()}
        avg["count"] = len(results)
        return avg


class PerformanceEvaluator:
    """性能评估: 响应时间 / 吞吐量 / 并发"""

    def evaluate(self, response_times: List[float]) -> Dict:
        if not response_times:
            return {}

        sorted_times = sorted(response_times)
        n = len(sorted_times)

        return {
            "count": n,
            "avg_time": round(sum(sorted_times) / n, 4),
            "p50_time": round(sorted_times[int(n * 0.5)], 4),
            "p90_time": round(sorted_times[int(n * 0.9)], 4),
            "p95_time": round(sorted_times[int(n * 0.95)], 4),
            "p99_time": round(sorted_times[int(n * 0.99)], 4),
            "max_time": round(sorted_times[-1], 4),
            "min_time": round(sorted_times[0], 4),
            "within_3s": sum(1 for t in sorted_times if t <= 3.0),
            "within_3s_ratio": round(sum(1 for t in sorted_times if t <= 3.0) / n, 4),
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # 演示
    ret_eval = RetrievalEvaluator(top_k=5)
    ret_results = [
        {"retrieved": [{"id": "1"}, {"id": "2"}, {"id": "3"}], "relevant_ids": {"1", "3", "5"}},
        {"retrieved": [{"id": "2"}, {"id": "4"}, {"id": "1"}], "relevant_ids": {"1", "5"}},
    ]
    print("检索评估:", json.dumps(ret_eval.evaluate_batch(ret_results), indent=2))

    qa_eval = QAEvaluator()
    qa_results = [
        {"answer": "注册资本7360万元", "reference": "注册资本7360万元",
         "ref_keywords": ["7360", "注册资本"], "context_chunks": ["注册资本7360万元"]},
    ]
    print("QA评估:", json.dumps(qa_eval.evaluate_batch(qa_results), indent=2))
