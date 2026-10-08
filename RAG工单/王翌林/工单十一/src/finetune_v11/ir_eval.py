# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-Embedding模型微调任务
src/finetune_v11/ir_eval.py —— 工单十一 检索评估器构造与指标对比（纯函数，可单测）

对应工单"创建评估器、微调前评估、微调后评估"要求：
  - 将 dev 问答对（+ 工单七 10 道真实金标题）转换为 sentence-transformers
    InformationRetrievalEvaluator 所需的 queries/corpus/relevant_docs；
  - 语料 = 问答对涉及的 chunk + 随机干扰 chunk（增大检索难度）；
  - 指标取 mrr@10 / ndcg@10 / accuracy@1 / accuracy@5，支撑
    "微调后检索效果优于微调前"的验收结论。
"""
import random
from typing import Any, Dict, Iterable, List, Set, Tuple

WORK_ORDER = "人工智能NLP-RAG-Embedding模型微调任务"

METRIC_KEYS = ("mrr@10", "ndcg@10", "map@100", "accuracy@1",
               "accuracy@3", "accuracy@5", "accuracy@10")


def build_ir_inputs(
        dev_pairs: List[Dict[str, Any]],
        corpus_chunks: List[Dict[str, Any]],
        gold_questions: List[Dict[str, Any]] = None,
        ) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, Set[str]]]:
    """工单十一：构造 InformationRetrievalEvaluator 输入

    dev_pairs: [{"query","chunk_id",...}]，chunk 级相关；
    gold_questions: 工单七真实题 [{"question","gold_docs":[...]}]，doc 级相关
    （语料中该文档全部 chunk 视为相关）；
    corpus_chunks: 检索语料 [{"chunk_id","doc_id","text",...}]。
    """
    queries: Dict[str, str] = {}
    corpus: Dict[str, str] = {}
    relevant: Dict[str, Set[str]] = {}

    for c in corpus_chunks:
        corpus[c["chunk_id"]] = c["text"]

    for i, p in enumerate(dev_pairs):
        qid = f"gen_{i}"
        queries[qid] = p["query"]
        if p["chunk_id"] in corpus:
            relevant.setdefault(qid, set()).add(p["chunk_id"])

    for i, g in enumerate(gold_questions or []):
        qid = f"v7_{g.get('id', i)}"
        queries[qid] = g["question"]
        gold_docs = set(g.get("gold_docs") or [])
        rel = {c["chunk_id"] for c in corpus_chunks if c["doc_id"] in gold_docs}
        if rel:
            relevant[qid] = rel

    # 过滤无语料相关的 query（避免评估器报错）
    queries = {qid: q for qid, q in queries.items() if qid in relevant}
    return queries, corpus, relevant


def pick_distractor_chunks(chunks: List[Dict[str, Any]],
                           exclude_ids: Set[str], n: int = 800,
                           seed: int = 42) -> List[Dict[str, Any]]:
    """工单十一：从语料中抽取干扰 chunk（排除已用于问答对生成的，防泄漏）"""
    pool = [c for c in chunks if c["chunk_id"] not in exclude_ids]
    rng = random.Random(seed)
    rng.shuffle(pool)
    return pool[:n]


def extract_metrics(evaluator_scores: Dict[str, Any],
                    prefix_keys: Iterable[str] = METRIC_KEYS) -> Dict[str, float]:
    """工单十一：从评估器输出中抽取关注指标（键名兼容带前缀形式）"""
    out: Dict[str, float] = {}
    for k, v in evaluator_scores.items():
        short = k.split("_")[-1] if "@" in k else k
        for want in prefix_keys:
            if short == want and isinstance(v, (int, float)):
                out[want] = round(float(v), 4)
    return out


def compare_metrics(before: Dict[str, float],
                    after: Dict[str, float]) -> List[Dict[str, Any]]:
    """工单十一：微调前后指标对比表（delta > 0 即提升）"""
    rows = []
    for k in METRIC_KEYS:
        if k in before and k in after:
            rows.append({"metric": k, "before": before[k], "after": after[k],
                         "delta": round(after[k] - before[k], 4)})
    return rows
