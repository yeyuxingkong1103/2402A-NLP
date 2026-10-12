# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-金融问答系统部署
RAG 核心：解析 PDF、BM25 检索、生成答案。
"""
import math
import os

import pymupdf

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


class RAGService:
    def __init__(self, pdf_paths):
        self.pdf_paths = pdf_paths
        self.docs = []
        self.doc_tokens = []
        self.df = {}
        self.n = 0
        self.avg_len = 0
        self.api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("DASHSCOPE_API_KEY")

    def build(self):
        for pdf in self.pdf_paths:
            if not os.path.exists(pdf):
                continue
            doc = pymupdf.open(pdf)
            text = "\n".join(p.get_text() for p in doc)
            doc.close()
            # 分块
            text = text.replace("\x00", " ").strip()
            for start in range(0, len(text), 500):
                self.docs.append(text[start:start + 500])
        self.doc_tokens = [tokenize(d) for d in self.docs]
        self.n = len(self.docs)
        for toks in self.doc_tokens:
            for t in set(toks):
                self.df[t] = self.df.get(t, 0) + 1
        self.avg_len = sum(len(t) for t in self.doc_tokens) / max(1, self.n)

    def _idf(self, term):
        df = self.df.get(term, 0)
        return math.log((self.n - df + 0.5) / (df + 0.5) + 1.0)

    def search(self, query, top_k=3):
        qt = tokenize(query)
        scored = []
        for i, toks in enumerate(self.doc_tokens):
            dl = len(toks)
            tf = {}
            for t in toks:
                tf[t] = tf.get(t, 0) + 1
            score = 0.0
            for term in set(qt):
                if term not in self.df:
                    continue
                f = tf.get(term, 0)
                if f == 0:
                    continue
                score += self._idf(term) * f * 2.5 / (f + 1.5 * (1 - 0.75 + 0.75 * dl / max(1, self.avg_len)))
            scored.append((i, score))
        scored.sort(key=lambda x: -x[1])
        return [self.docs[i] for i, s in scored if s > 0][:top_k]

    def answer(self, query):
        hits = self.search(query)
        ctx = "\n".join(hits)
        if self.api_key:
            return self._llm_answer(query, ctx)
        return ctx.strip()  # 降级：返回检索上下文

    def _llm_answer(self, query, ctx):
        import urllib.request
        import json
        base_url = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
        model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
        prompt = f"请依据参考上下文回答，答案准确简洁。\n参考上下文：\n{ctx}\n\n问题：{query}"
        payload = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.1}
        req = urllib.request.Request(base_url + "/chat/completions",
                                     data=json.dumps(payload).encode("utf-8"),
                                     headers={"Content-Type": "application/json",
                                              "Authorization": "Bearer " + self.api_key})
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))["choices"][0]["message"]["content"]
