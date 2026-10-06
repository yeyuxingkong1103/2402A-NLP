# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-功能测试及评估
评估框架：实现 RAGAS 风格的四项指标（简化离线实现）——
  faithfulness         忠实度（答案是否基于检索上下文）
  answer_relevance     答案相关性（答案与问题的相关度）
  context_precision    上下文精度（相关上下文排序是否靠前）
  context_recall       上下文召回（参考答案信息是否被上下文覆盖）
"""
from retriever import tokenize


def _overlap(a: str, b: str) -> float:
    ta = set(tokenize(a))
    tb = set(tokenize(b))
    if not ta:
        return 0.0
    return len(ta & tb) / len(ta)


def faithfulness(answer, contexts) -> float:
    """答案中能在上下文找到支撑的比例。"""
    if not answer or not contexts:
        return 0.0
    ctx = " ".join(contexts)
    return _overlap(answer, ctx)


def answer_relevance(question, answer) -> float:
    """答案与问题的语义重叠度（简化）。"""
    if not answer:
        return 0.0
    return _overlap(question, answer)


def context_precision(question, contexts, reference) -> float:
    """参考信息是否在靠前的上下文中出现（按命中加权）。"""
    if not reference or not contexts:
        return 0.0
    rel = [1.0 if _overlap(c, reference) > 0 else 0.0 for c in contexts]
    if not any(rel):
        return 0.0
    weighted = sum(r / (i + 1) for i, r in enumerate(rel))
    return weighted / sum(rel)


def context_recall(question, contexts, reference) -> float:
    """参考答案关键信息被上下文覆盖的比例。"""
    if not reference:
        return 0.0
    ctx = " ".join(contexts)
    return _overlap(reference, ctx)


def evaluate(question, answer, contexts, reference) -> dict:
    ctx_texts = [c for c, _ in contexts]
    return {
        "faithfulness": round(faithfulness(answer, ctx_texts), 3),
        "answer_relevance": round(answer_relevance(question, answer), 3),
        "context_precision": round(context_precision(question, ctx_texts, reference), 3),
        "context_recall": round(context_recall(question, ctx_texts, reference), 3),
    }
