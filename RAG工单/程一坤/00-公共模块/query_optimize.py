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
import re            # 从 LLM 输出中提取 JSON 数组
import json          # 解析 LLM 输出的查询变体数组
import jieba.analyse  # jieba 关键词抽取（TF-IDF 算法），用于关键词重排

from ollama_client import client  # Ollama 单例：做查询扩展生成

# 查询扩展 Prompt：只输出 JSON 数组
# 强制"只输出 JSON"是为了后续能用正则+json.loads 可靠解析
EXPAND_PROMPT = """你是一个检索查询改写器。请把下面的问题改写成 {n} 个适合在招股说明书中检索的等价问法。
要求：
1. 语义完全等价，只是换用同义词/文档常用表述（例："军用领域的收入"可改写为"军品收入"、"军用业务收入"）；
2. 保留公司名称等关键实体；
3. 只输出一个 JSON 字符串数组，不要其他内容。

【问题】{question}

【JSON数组】"""


def expand_query(question, n=3, retries=2):
    """LLM 查询扩展，返回 [原问题, 变体1, 变体2, ...]（失败时返回原问题单路）"""
    variants = [question]  # 原问题永远保留在首位，LLM 失败也不影响基本检索
    for i in range(retries):
        try:
            # temperature=0.0：改写要确定性、语义等价，禁止发散
            raw = client.generate(
                EXPAND_PROMPT.format(n=n, question=question),
                temperature=0.0, num_predict=200)
            # re.S 让 . 匹配换行：LLM 可能把 JSON 数组输出成多行
            # 贪婪匹配最外层 [ ... ]，截掉 LLM 可能附带的解释文字
            m = re.search(r"\[.*\]", raw, re.S)
            if m:
                arr = json.loads(m.group(0))  # 解析失败会抛异常进入 except 重试
                # str(v) 强转防止 LLM 输出数字/嵌套；strip() 去空白；[:n] 截断多余变体
                variants += [str(v).strip() for v in arr if str(v).strip()][:n]
            break  # 成功拿到结果（或拿到了但解析失败）即跳出重试循环
        except Exception:
            # LLM 输出格式不合法（如带注释/单引号）时重试
            continue
    # dict.fromkeys 利用字典键唯一且保序的特性去重，变体重复时不多路检索浪费算力
    return list(dict.fromkeys(variants))  # 去重保序


def multi_retrieve(store, variants, top_k=5, fetch_k=8):
    """多路召回融合：以单路最大向量相似度为主，RRF做多路一致性小幅加分。
    设计原因：纯RRF会稀释"原问法下单路强命中"的真值分块（实测退步教训），
    融合分 = max(向量相似度) + 0.05*RRF累计值。
    """
    fused = {}  # key=块文本（文本即唯一标识），value=该块的融合信息
    for v in variants:
        # 每个变体独立检索 fetch_k 条（fetch_k>top_k：多召一些再融合筛选）
        for rank, h in enumerate(store.search(v, top_k=fetch_k)):
            key = h["text"]
            if key not in fused:
                # 首次命中：vec 记录向量相似度，rrf 从 0 开始累计
                fused[key] = {**h, "vec": h["score"], "rrf": 0.0}
            else:
                # 多路命中同一块：vec 取各路最大值（max 比 avg 更能保留单路强信号）
                fused[key]["vec"] = max(fused[key]["vec"], h["score"])
            # RRF 标准公式：1/(k+rank)，k=60 是业界经验值，平滑头部排名差异
            fused[key]["rrf"] += 1.0 / (60 + rank + 1)
    # 融合排序：向量相似度为主、RRF 为辅（0.05 权重是多路一致性的小加分），
    # 负号实现降序，截取 top_k 返回
    ranked = sorted(fused.values(),
                    key=lambda x: -(x["vec"] + 0.05 * x["rrf"]))[:top_k]
    for h in ranked:
        # 把融合分写回 score 字段，下游（重排/问答）统一用 score
        h["score"] = h["vec"] + 0.05 * h["rrf"]
    return ranked


def keyword_rerank(query, hits, boost=0.05):
    """关键词重排：jieba抽取query关键词，剔除非区分性词（出现在过半候选中的词，
    如公司全称），仅对区分性关键词命中的分块加权。"""
    # extract_tags 基于 TF-IDF 抽前 10 个关键词，withWeight=True 返回 (词, 权重)
    kws = [w for w, _ in jieba.analyse.extract_tags(query, topK=10, withWeight=True)]
    if not kws or not hits:
        return hits  # 无关键词或无候选时原样返回
    n = len(hits)
    # 过滤在过半候选中都出现的"非区分性"关键词
    # （如公司全称，几乎每个分块都有，加权只会引入噪声）
    disc = [kw for kw in kws
            if sum(1 for h in hits if kw in h["text"]) <= n * 0.5]
    if not disc:
        return hits  # 关键词全无区分性时不重排
    for h in hits:
        # 统计该块命中的区分性关键词个数，每命中一个加 boost 分
        overlap = sum(1 for kw in disc if kw in h["text"])
        h["score"] = h["score"] + boost * overlap
    # 按加分后的总分降序重排
    return sorted(hits, key=lambda x: -x["score"])
