# -*- coding: utf-8 -*-
"""
混合检索模块（工单06，供工单03/05/07复用）
工单编号：人工智能NLP-RAG-混合检索任务
功能：
  1. BM25 全文检索（jieba分词 + 倒排索引），支持布尔查询(AND/OR/NOT)、
     短语匹配("...")、模糊匹配（编辑距离≤1的词项近似）；
  2. 多检索策略：向量检索 / 全文检索 / 混合检索（权重可配）；
  3. 三种重排器：LLM重排器、TF-IDF重排器、用户反馈自适应重排器。
"""
import os
import re
import json
import difflib

import jieba
from rank_bm25 import BM25Okapi

from config import INDEX_DIR


def tokenize(text):
    """jieba 搜索引擎模式分词"""
    return list(jieba.cut_for_search(str(text)))


class BM25Index:
    """BM25 全文检索索引（基于倒排思想，rank_bm25 实现）"""

    def __init__(self):
        self.texts = []
        self.metadatas = []
        self._bm25 = None
        self._tokenized = []

    # ── 构建 / 加载 ────────────────────────────────────────
    def build(self, texts, metadatas):
        self.texts = list(texts)
        self.metadatas = list(metadatas)
        print(f"[BM25] 分词 {len(self.texts)} 条 ...")
        self._tokenized = [tokenize(t) for t in self.texts]
        self._bm25 = BM25Okapi(self._tokenized)
        return self

    def save(self, name):
        os.makedirs(INDEX_DIR, exist_ok=True)
        data = {"texts": self.texts, "metadatas": self.metadatas}
        with open(os.path.join(INDEX_DIR, f"{name}.bm25.json"), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        print(f"[BM25] 已保存 {name}")

    @classmethod
    def load(cls, name, auto_build=True):
        path = os.path.join(INDEX_DIR, f"{name}.bm25.json")
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        idx = cls()
        idx.build(data["texts"], data["metadatas"]) if auto_build else None
        if not auto_build:
            idx.texts, idx.metadatas = data["texts"], data["metadatas"]
        return idx

    # ── 查询解析：布尔 / 短语 / 模糊 ────────────────────────
    @staticmethod
    def parse_query(query):
        """解析查询：提取短语("...")、布尔操作符(AND/OR/NOT)、普通词项"""
        phrases = re.findall(r'"([^"]+)"|"(.+?)"', query)
        phrases = [a or b for a, b in phrases]
        q = re.sub(r'"[^"]*"', " ", query)
        must, should, must_not = [], [], []
        tokens = [t for t in tokenize(q) if t.strip()]
        i = 0
        while i < len(tokens):
            t = tokens[i].upper()
            if t == "AND" and i + 1 < len(tokens):
                must.append(tokens[i + 1]); i += 2
            elif t == "OR" and i + 1 < len(tokens):
                should.append(tokens[i + 1]); i += 2
            elif t == "NOT" and i + 1 < len(tokens):
                must_not.append(tokens[i + 1]); i += 2
            else:
                if t not in ("AND", "OR", "NOT"):
                    should.append(tokens[i])
                i += 1
        return {"phrases": phrases, "must": must, "should": should, "must_not": must_not}

    @staticmethod
    def _fuzzy_in(token, text):
        """模糊匹配：词项或其编辑距离≤1的近似词出现在文本中"""
        if token in text:
            return True
        # 从文本中抽相似词做编辑距离近似（限制长度防止全表扫描过慢）
        for w in set(re.findall(re.escape(token[0]) + r"\w{0,%d}" % (len(token) + 1), text)):
            if difflib.SequenceMatcher(None, token, w).ratio() >= 0.85:
                return True
        return False

    def search(self, query, top_k=10, fuzzy=True):
        """全文检索：BM25 打分 + 布尔/短语过滤"""
        parsed = self.parse_query(query)
        base_query = " ".join(parsed["should"] + parsed["must"]) or query
        scores = self._bm25.get_scores(tokenize(base_query))
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[: top_k * 6]

        results = []
        for i in order:
            text = self.texts[i]
            if scores[i] <= 0:
                continue
            # 布尔约束
            if any(w not in text for w in parsed["must"]):
                continue
            if any(w in text for w in parsed["must_not"]):
                continue
            if parsed["should"] and fuzzy:
                if not any(self._fuzzy_in(w, text) for w in parsed["should"]):
                    continue
            elif parsed["should"] and not any(w in text for w in parsed["should"]):
                continue
            # 短语约束
            if parsed["phrases"] and not all(p in text for p in parsed["phrases"]):
                continue
            results.append({"text": text, "page": self.metadatas[i].get("page"),
                            "source": self.metadatas[i].get("source"),
                            "score": float(scores[i])})
            if len(results) >= top_k:
                break
        return results


# ── 混合检索 ────────────────────────────────────────────────
def _norm(scores):
    """min-max 归一化到 0~1"""
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    return [(s - lo) / (hi - lo + 1e-8) for s in scores]


def hybrid_search(vector_store, bm25_index, query,
                  mode="hybrid", vec_w=0.7, ft_w=0.3, top_k=5, fetch_k=20):
    """统一检索入口
    mode: vector=仅向量 | fulltext=仅BM25全文 | hybrid=加权混合
    融合算法：各路分数 min-max 归一化后加权求和（可配权重）
    """
    if mode == "vector":
        return vector_store.search(query, top_k=top_k)
    if mode == "fulltext":
        return bm25_index.search(query, top_k=top_k)

    v_hits = vector_store.search(query, top_k=fetch_k)
    f_hits = bm25_index.search(query, top_k=fetch_k)

    fused = {}
    for h, s in zip(v_hits, _norm([h["score"] for h in v_hits])):
        key = h["text"]
        fused[key] = {**h, "vs": s, "fs": 0.0}
    for h, s in zip(f_hits, _norm([h["score"] for h in f_hits])):
        key = h["text"]
        if key in fused:
            fused[key]["fs"] = s
        else:
            fused[key] = {**h, "vs": 0.0, "fs": s}

    for h in fused.values():
        h["score"] = vec_w * h["vs"] + ft_w * h["fs"]
    ranked = sorted(fused.values(), key=lambda x: -x["score"])[:top_k]
    return ranked


# ── 三种重排器 ──────────────────────────────────────────────
LLM_RERANK_PROMPT = """你是检索结果重排器。请根据与问题的相关性，对下列片段重新排序。
只输出排序后的编号列表（JSON数组，如 [3,1,2]），不要解释。

【问题】{query}

【候选片段】
{candidates}

【排序结果】"""


def llm_rerank(query, hits, client, top_m=8):
    """基于LLM的重排器：让LLM对候选片段按相关性排序"""
    cands = hits[:top_m]
    cand_text = "\n".join(f"[{i+1}] {h['text'][:150]}" for i, h in enumerate(cands))
    try:
        raw = client.generate(LLM_RERANK_PROMPT.format(query=query, candidates=cand_text),
                              temperature=0.0, num_predict=80)
        import json as _json
        m = re.search(r"\[[\d,\s]+\]", raw)
        order = _json.loads(m.group(0)) if m else []
        reranked = [cands[i - 1] for i in order if 1 <= i <= len(cands)]
        reranked += [h for h in cands if h not in reranked]
        return reranked + hits[top_m:]
    except Exception:
        return hits


def tfidf_rerank(query, hits, top_m=10):
    """基于TF-IDF的重排器：候选片段集上构建TF-IDF向量，与query余弦相似度重排"""
    cands = hits[:top_m]
    docs = [tokenize(h["text"]) for h in cands] + [tokenize(query)]
    # 构建 IDF
    import math
    N = len(docs)
    df = {}
    for d in docs:
        for w in set(d):
            df[w] = df.get(w, 0) + 1
    idf = {w: math.log((N + 1) / (c + 1)) + 1 for w, c in df.items()}

    def tfidf_vec(tokens):
        tf = {}
        for w in tokens:
            tf[w] = tf.get(w, 0) + 1
        return {w: (c / len(tokens)) * idf.get(w, 0) for w, c in tf.items()}

    vecs = [tfidf_vec(d) for d in docs]
    qv = vecs[-1]

    def cos(a, b):
        dot = sum(a.get(w, 0) * b.get(w, 0) for w in set(a) | set(b))
        na = math.sqrt(sum(v * v for v in a.values()))
        nb = math.sqrt(sum(v * v for v in b.values()))
        return dot / (na * nb + 1e-8)

    scored = sorted(range(len(cands)), key=lambda i: -cos(vecs[i], qv))
    reranked = [cands[i] for i in scored]
    return reranked + hits[top_m:]


FEEDBACK_FILE = os.path.join(INDEX_DIR, "feedback.json")


class FeedbackReranker:
    """基于用户反馈的自适应重排器：
    用户采纳过的来源页（反馈记录）在后续检索中获得加权，
    反馈随使用累积，实现自适应个性化排序。"""

    def __init__(self, boost=0.15):
        self.boost = boost
        if os.path.exists(FEEDBACK_FILE):
            with open(FEEDBACK_FILE, encoding="utf-8") as f:
                self.fb = json.load(f)
        else:
            self.fb = {}  # {"来源名|页码": 次数}

    def rerank(self, hits):
        def key(h):
            k = f"{h.get('source')}|{h.get('page')}"
            return -h["score"] - self.boost * self.fb.get(k, 0)
        return sorted(hits, key=key)

    def record(self, hits):
        """把本轮被采纳的来源记入反馈"""
        for h in hits[:3]:
            k = f"{h.get('source')}|{h.get('page')}"
            self.fb[k] = self.fb.get(k, 0) + 1
        os.makedirs(INDEX_DIR, exist_ok=True)
        with open(FEEDBACK_FILE, "w", encoding="utf-8") as f:
            json.dump(self.fb, f, ensure_ascii=False)


RERANKERS = {"llm": llm_rerank, "tfidf": tfidf_rerank}


def apply_rerank(rerank_name, query, hits, client=None, top_k=5):
    """统一重排入口: rerank_name ∈ {none, llm, tfidf, feedback}"""
    if rerank_name == "none" or not hits:
        return hits
    if rerank_name == "feedback":
        return FeedbackReranker().rerank(hits)[:top_k]
    fn = RERANKERS.get(rerank_name)
    if fn is None:
        return hits
    args = (query, hits) if rerank_name == "tfidf" else (query, hits, client)
    return fn(*args)[:top_k]
