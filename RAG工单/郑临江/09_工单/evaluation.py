# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-Graph RAG优化任务
评估模块：RAGAS 上下文精度（context precision）与上下文召回（context recall）。
"""
try:
    import jieba
except ImportError:
    jieba = None


def tokenize(text):
    text = text.lower()
    if jieba is not None:
        return [t.strip() for t in jieba.cut(text) if t.strip()]
    chars = [c for c in text if not c.isspace()]
    return chars + [chars[i] + chars[i + 1] for i in range(len(chars) - 1)]


def _overlap(a, b):
    ta, tb = set(tokenize(a)), set(tokenize(b))
    return len(ta & tb) / len(ta) if ta else 0.0


def context_precision(contexts, reference):
    """相关上下文是否被排在靠前位置。"""
    if not reference or not contexts:
        return 0.0
    rel = [1.0 if _overlap(c, reference) > 0.05 else 0.0 for c in contexts]
    if not any(rel):
        return 0.0
    return sum(r / (i + 1) for i, r in enumerate(rel)) / sum(rel)


def context_recall(contexts, reference):
    """参考答案信息被上下文覆盖的比例。"""
    if not reference:
        return 0.0
    ctx = " ".join(contexts)
    return _overlap(reference, ctx)


def evaluate(contexts, reference):
    return {
        "context_precision": round(context_precision(contexts, reference), 3),
        "context_recall": round(context_recall(contexts, reference), 3),
    }
