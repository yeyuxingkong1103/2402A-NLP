# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG项目-Embedding模型微调任务
"""
信息检索评估器：在金融年报评估集上评估 Embedding 模型的检索质量。

指标：
    Recall@1 / @3 / @5 / @10
    MRR@10（平均倒数排名）
    Top1 准确率
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder  # noqa: F401  (保留接口一致性)


class IREvaluator:
    """信息检索评估器。"""

    def __init__(self, eval_pairs, corpus_texts, name: str = "finance-eval"):
        """
        Args:
            eval_pairs: [{"query": str, "positive": str}, ...]
            corpus_texts: List[str] 全部候选段落
        """
        self.queries = [p["query"] for p in eval_pairs]
        self.positives = [p["positive"] for p in eval_pairs]
        self.corpus = corpus_texts
        self.name = name

        # 正例段落在 corpus 中的索引
        text_to_idx = {t: i for i, t in enumerate(corpus_texts)}
        self.pos_idx = [text_to_idx[p] for p in self.positives]

    @staticmethod
    def _recall_at_k(rank_lists, pos_idx, k):
        hits = sum(1 for ranks, gold in zip(rank_lists, pos_idx) if gold in ranks[:k])
        return hits / len(pos_idx)

    @staticmethod
    def _mrr(rank_lists, pos_idx, cutoff=10):
        total = 0.0
        for ranks, gold in zip(rank_lists, pos_idx):
            for rank, idx in enumerate(ranks[:cutoff], start=1):
                if idx == gold:
                    total += 1.0 / rank
                    break
        return total / len(pos_idx)

    def evaluate(self, model: SentenceTransformer, verbose: bool = True, encode_corpus=None):
        """运行评估，返回指标字典。"""
        # 编码语料
        if encode_corpus is not None:
            corpus_emb = encode_corpus
        else:
            corpus_emb = model.encode(
                self.corpus, batch_size=32, convert_to_numpy=True,
                normalize_embeddings=True, show_progress_bar=verbose,
            )
        # 编码查询
        query_emb = model.encode(
            self.queries, batch_size=32, convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=False,
        )

        # 余弦相似度（已归一化 → 点积）
        sims = query_emb @ corpus_emb.T

        # 按相似度降序排名
        rank_lists = np.argsort(-sims, axis=1).tolist()

        metrics = {
            "Recall@1": round(self._recall_at_k(rank_lists, self.pos_idx, 1), 4),
            "Recall@3": round(self._recall_at_k(rank_lists, self.pos_idx, 3), 4),
            "Recall@5": round(self._recall_at_k(rank_lists, self.pos_idx, 5), 4),
            "Recall@10": round(self._recall_at_k(rank_lists, self.pos_idx, 10), 4),
            "MRR@10": round(self._mrr(rank_lists, self.pos_idx, 10), 4),
            "Top1准确率": round(self._recall_at_k(rank_lists, self.pos_idx, 1), 4),
        }

        if verbose:
            print(f"\n[{self.name}] 检索评估结果（{len(self.queries)} 条查询 / {len(self.corpus)} 个段落）：")
            for k, v in metrics.items():
                print(f"  {k:10s}: {v:.4f}")

        return metrics
