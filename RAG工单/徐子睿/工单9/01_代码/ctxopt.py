# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-基于Graph RAG实现金融问答 | 人工智能NLP-RAG-Graph RAG优化任务
# 模块：ctxopt —— Graph RAG 优化（工单 09）之「证据级上下文压缩」
# 说明：检索回来的块往往整页/整段命中，包含大量与问题无关的叙述，会稀释上下文精度。
#       这里只保留命中「公司 / 指标 / 年份 / 金额」的窗口片段（证据句），
#       在不丢证据的前提下缩短上下文 → 提升 context_precision，同时降低生成时延。
import re

UNIT_RE = re.compile(r"\d[\d,]*\.?\d*\s*(?:亿元|万元|元|%|％|百分点)")
SPLIT_RE = re.compile(r"\s+")


def evidence_spans(text, keys, win=90, cap=420):
    """定位证据窗口：命中关键词或「数字+单位」的位置，各向外扩 win 字符。"""
    t = SPLIT_RE.sub(" ", text or "")
    spans = []
    for k in keys:
        if not k:
            continue
        for m in re.finditer(re.escape(k), t):
            spans.append((max(0, m.start() - win), min(len(t), m.end() + win)))
    for m in UNIT_RE.finditer(t):
        spans.append((max(0, m.start() - win), min(len(t), m.end() + win)))
    if not spans:
        return [], t
    spans.sort()
    merged = [list(spans[0])]
    for a, b in spans[1:]:
        if a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return merged, t


def compress_context(text, keys, win=90, cap=420):
    """工单 09：把单块文本压缩为「证据窗口」，无命中时退化为截断。"""
    spans, t = evidence_spans(text, keys, win=win, cap=cap)
    if not spans:
        return t[:cap]
    out = " … ".join(t[a:b] for a, b in spans)
    return out[:cap]


def compress_hits(hits, entities, win=90, cap=420):
    """对检索结果整体做压缩；entities 为 {"公司": [...], "指标": [...], "年份": [...]}。"""
    keys = list((entities or {}).get("公司") or []) + \
           list((entities or {}).get("指标") or []) + \
           list((entities or {}).get("年份") or [])
    out = []
    for h in hits:
        h2 = dict(h)
        h2["text"] = compress_context(h.get("text", ""), keys, win=win, cap=cap)
        h2["compressed"] = True
        out.append(h2)
    return out
