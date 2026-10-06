# -*- coding: utf-8 -*-
"""
检索优化模块（工单02，供工单05/06/12复用）
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化
说明：三项优化手段——
  1. LLM 查询扩展：把口语问法改写为文档表述（"军用领域的收入"→"军品收入"），
     多路召回解决"问法与文档用词不一致导致召回失败"的问题；
  2. 多路召回 RRF 融合：多个查询变体各自检索，用倒数排名融合，兼顾各问法；
  3. 关键词重排：query 关键词与分块字面重叠加分，缓解向量语义漂移。
"""
import re
import json
import jieba.analyse

from ollama_client import client

# 查询扩展 Prompt：只输出 JSON 数组
EXPAND_PROMPT = """你是一个检索查询改写器。请把下面的问题改写成 {n} 个适合在招股说明书中检索的等价问法。
要求：
1. 语义完全等价，只是换用同义词/文档常用表述（例："军用领域的收入"可改写为"军品收入"、"军用业务收入"）；
2. 保留公司名称等关键实体；
3. 只输出一个 JSON 字符串数组，不要其他内容。

【问题】{question}

【JSON数组】"""


def expand_query(question, n=3, retries=2):
    """LLM 查询扩展，返回 [原问题, 变体1, 变体2, ...]（失败时返回原问题单路）"""
    variants = [question]
    for i in range(retries):
        try:
            raw = client.generate(
                EXPAND_PROMPT.format(n=n, question=question),
                temperature=0.0, num_predict=200)
            m = re.search(r"\[.*\]", raw, re.S)
            if m:
                arr = json.loads(m.group(0))
                variants += [str(v).strip() for v in arr if str(v).strip()][:n]
            break
        except Exception:
            continue
    return list(dict.fromkeys(variants))  # 去重保序


def multi_retrieve(store, variants, top_k=5, fetch_k=8):
    """多路召回融合：以单路最大向量相似度为主，RRF做多路一致性小幅加分。
    设计原因：纯RRF会稀释"原问法下单路强命中"的真值分块（实测退步教训），
    融合分 = max(向量相似度) + 0.05*RRF累计值。
    """
    fused = {}
    for v in variants:
        for rank, h in enumerate(store.search(v, top_k=fetch_k)):
            key = h["text"]
            if key not in fused:
                fused[key] = {**h, "vec": h["score"], "rrf": 0.0}
            else:
                fused[key]["vec"] = max(fused[key]["vec"], h["score"])
            fused[key]["rrf"] += 1.0 / (60 + rank + 1)
    ranked = sorted(fused.values(),
                    key=lambda x: -(x["vec"] + 0.05 * x["rrf"]))[:top_k]
    for h in ranked:
        h["score"] = h["vec"] + 0.05 * h["rrf"]
    return ranked


def keyword_rerank(query, hits, boost=0.05):
    """关键词重排：jieba抽取query关键词，剔除非区分性词（出现在过半候选中的词，
    如公司全称），仅对区分性关键词命中的分块加权。"""
    kws = [w for w, _ in jieba.analyse.extract_tags(query, topK=10, withWeight=True)]
    if not kws or not hits:
        return hits
    n = len(hits)
    # 过滤在过半候选中都出现的"非区分性"关键词
    disc = [kw for kw in kws
            if sum(1 for h in hits if kw in h["text"]) <= n * 0.5]
    if not disc:
        return hits
    for h in hits:
        overlap = sum(1 for kw in disc if kw in h["text"])
        h["score"] = h["score"] + boost * overlap
    return sorted(hits, key=lambda x: -x["score"])
