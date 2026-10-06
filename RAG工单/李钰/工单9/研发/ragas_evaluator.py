# -*- coding: utf-8 -*-
"""
RAGAS 风格评估器 - context precision / context recall
工单编号: 人工智能 NLP-RAG-Graph RAG 优化任务
"""
import json, os, logging, math
from typing import List, Dict, Set

logger = logging.getLogger(__name__)


class RAGASEvaluator:
    """
    RAGAS 风格评估器

    核心指标:
      - Context Precision: 检索到的上下文中有多少是相关的
      - Context Recall: 所有相关上下文中有多少被检索到
      - Faithfulness: 答案中有多少可从上下文推断
      - Answer Relevance: 答案与问题的相关程度
    """

    def __init__(self):
        pass

    def evaluate_context_precision(self, retrieved_contexts: List[str],
                                    reference_contexts: List[str]) -> float:
        """
        Context Precision = |relevant ∩ retrieved| / |retrieved|

        简化实现: 用 Jaccard + 关键词匹配
        """
        if not retrieved_contexts:
            return 0.0

        retrieved_set = self._tokenize_set(" ".join(retrieved_contexts))
        reference_set = self._tokenize_set(" ".join(reference_contexts))

        if not retrieved_set:
            return 0.0

        intersection = retrieved_set & reference_set
        precision = len(intersection) / len(retrieved_set)
        return round(precision, 4)

    def evaluate_context_recall(self, retrieved_contexts: List[str],
                                 reference_contexts: List[str]) -> float:
        """
        Context Recall = |relevant ∩ retrieved| / |relevant|
        """
        if not reference_contexts:
            return 1.0

        retrieved_set = self._tokenize_set(" ".join(retrieved_contexts))
        reference_set = self._tokenize_set(" ".join(reference_contexts))

        if not reference_set:
            return 1.0

        intersection = retrieved_set & reference_set
        recall = len(intersection) / len(reference_set)
        return round(recall, 4)

    def evaluate_context_precision_recall(self, retrieved_contexts: List[str],
                                           reference_contexts: List[str]) -> Dict:
        """P@K 和 R@K 同时计算"""
        precision = self.evaluate_context_precision(retrieved_contexts, reference_contexts)
        recall = self.evaluate_context_recall(retrieved_contexts, reference_contexts)
        # F1
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        return {
            "context_precision": precision,
            "context_recall": recall,
            "context_f1": round(f1, 4),
        }

    def evaluate_from_keywords(self, retrieved_contexts: List[str],
                                 ref_keywords: List[str]) -> Dict:
        """
        用参考关键词评估 (当没有完整参考上下文时)

        Context Precision = 命中 ref_keywords 数 / 总 ref_keywords 在 retrieved 中出现数
        Context Recall = 命中 ref_keywords 数 / ref_keywords 总数
        """
        joined_retrieved = " ".join(retrieved_contexts) if retrieved_contexts else ""

        # 计算每个关键词的命中情况
        hit_keywords = [kw for kw in ref_keywords if kw in joined_retrieved]

        if not ref_keywords:
            return {"context_precision": 0.0, "context_recall": 0.0, "context_f1": 0.0}

        recall = len(hit_keywords) / len(ref_keywords)

        # precision: 用关键词在 retrieved 中的覆盖率估算
        retrieved_kw_count = sum(1 for kw in ref_keywords if kw in joined_retrieved)
        precision = retrieved_kw_count / max(len(retrieved_contexts), 1)
        precision = min(precision, 1.0)

        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        return {
            "context_precision": round(precision, 4),
            "context_recall": round(recall, 4),
            "context_f1": round(f1, 4),
            "hit_keywords": hit_keywords,
            "total_keywords": len(ref_keywords),
        }

    def evaluate_faithfulness(self, answer: str, contexts: List[str]) -> float:
        """
        Faithfulness: 答案中有多少内容可从检索上下文推断

        简化: 分句 → 检查每分句的关键子串是否在上下文中
        """
        if not answer or not contexts:
            return 0.0

        joined_context = " ".join(contexts)
        answer_sentences = [s.strip() for s in re.split(r"[。！？.!?]", answer) if s.strip()]

        if not answer_sentences:
            return 0.0

        faithful_count = 0
        for sent in answer_sentences:
            # 取关键片段 (前 10 字符)
            key = sent[:8]
            if key in joined_context or any(key[:5] in c for c in contexts):
                faithful_count += 1

        return round(faithful_count / len(answer_sentences), 4)

    def evaluate_batch(self, results: List[Dict]) -> Dict:
        """批量评估"""
        all_p, all_r, all_f1, all_faith = [], [], [], []
        for r in results:
            if "ref_keywords" in r:
                m = self.evaluate_from_keywords(
                    r.get("retrieved_contexts", []), r["ref_keywords"])
            else:
                m = self.evaluate_context_precision_recall(
                    r.get("retrieved_contexts", []), r.get("reference_contexts", []))
            all_p.append(m["context_precision"])
            all_r.append(m["context_recall"])
            all_f1.append(m.get("context_f1", 0))
            all_faith.append(self.evaluate_faithfulness(
                r.get("answer", ""), r.get("retrieved_contexts", [])))

        n = len(results)
        return {
            "avg_context_precision": round(sum(all_p) / max(n, 1), 4),
            "avg_context_recall": round(sum(all_r) / max(n, 1), 4),
            "avg_context_f1": round(sum(all_f1) / max(n, 1), 4),
            "avg_faithfulness": round(sum(all_faith) / max(n, 1), 4),
            "count": n,
        }

    def _tokenize_set(self, text: str) -> Set[str]:
        try:
            import jieba
            return set(t for t in jieba.cut(text) if t.strip())
        except ImportError:
            return set(c for c in text if c.strip())


import re  # noqa: E402

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ev = RAGASEvaluator()

    retrieved = ["武汉力源科技有限公司是武汉力源信息技术股份有限公司的控股股东(持股35%)"]
    reference = ["武汉力源科技有限公司是武汉力源的控股股东"]
    kws = ["武汉力源科技", "35", "控股股东"]

    print(f"P/R (上下文): {ev.evaluate_context_precision_recall(retrieved, reference)}")
    print(f"P/R (关键词): {ev.evaluate_from_keywords(retrieved, kws)}")
    print(f"Faithfulness: {ev.evaluate_faithfulness('控股股东是武汉力源科技,持股35%', retrieved)}")
