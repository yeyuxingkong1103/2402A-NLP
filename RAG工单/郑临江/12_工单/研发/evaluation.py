# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG项目-LightRAG优化
RAGAS 风格评估指标：faithfulness（忠实度）、answer_relevance（答案相关性）、
context_precision（上下文精确率）、context_recall（上下文召回率）。
"""
from rag import tokenize


def _tokens(text):
    return set(tokenize(text))


def faithfulness(answer, contexts):
    if not answer:
        return 0.0
    atok = _tokens(answer)
    ctext = "".join(contexts)
    hit = sum(1 for t in atok if t in ctext)
    return round(hit / max(len(atok), 1), 3)


def answer_relevance(answer, question):
    if not answer:
        return 0.0
    qtok = _tokens(question)
    atok = _tokens(answer)
    hit = sum(1 for t in atok if t in qtok)
    return round(hit / max(len(qtok), 1), 3)


def context_precision(contexts, question):
    if not contexts:
        return 0.0
    qtok = _tokens(question)
    scores = [len(qtok & _tokens(c)) / max(len(qtok), 1) for c in contexts]
    return round(sum(scores) / len(scores), 3)


def context_recall(contexts, question):
    if not contexts:
        return 0.0
    qtok = _tokens(question)
    ctext = "".join(contexts)
    hit = sum(1 for t in qtok if t in ctext)
    return round(hit / max(len(qtok), 1), 3)


def evaluate(result, question):
    ctx = result.get("contexts", [])
    ans = result.get("answer", "")
    return {
        "faithfulness": faithfulness(ans, ctx),
        "answer_relevance": answer_relevance(ans, question),
        "context_precision": context_precision(ctx, question),
        "context_recall": context_recall(ctx, question),
    }
