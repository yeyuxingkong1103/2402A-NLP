# -*- coding: utf-8 -*-
# 工单编号：人工智能 NLP-RAG-基于 PDF 文档的问答系统优化
# 模块：kb —— 知识库（内嵌向量库 + BM25 索引）
# 说明：块向量以 numpy 常驻内存做余弦检索（单文档万级块毫秒级），BM25 走自实现倒排；
#       两者一并持久化到 data/index/，启动即加载，无需外部向量服务。

import os
import json

import numpy as np

from config import CHUNKS_PATH, EMB_PATH, BM25_PATH, META_PATH, INDEX_DIR
from bm25 import BM25, tokenize
import llm


class KB:
    def __init__(self):
        self.chunks = []
        self.emb = None            # (N, D) float32, L2 归一化
        self.bm25 = BM25()
        self.token_lists = []
        self.meta = {}

    # ---------- 构建 ----------
    def build(self, chunks, batch=16, progress=None):
        self.chunks = chunks
        texts = [c["text"] for c in chunks]
        embs = []
        for i in range(0, len(texts), batch):
            part = texts[i:i + batch]
            embs.extend(llm.embed(part))
            if progress:
                progress(min(i + batch, len(texts)), len(texts))
        self.emb = np.asarray(embs, dtype=np.float32)
        # L2 归一化 -> 点积即余弦
        norms = np.linalg.norm(self.emb, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.emb /= norms
        self.token_lists = [tokenize(t) for t in texts]
        self.bm25.fit_tokens(self.token_lists)
        self.meta = {"count": len(chunks), "dim": int(self.emb.shape[1])}
        return self

    def save(self):
        os.makedirs(INDEX_DIR, exist_ok=True)
        with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        np.save(EMB_PATH, self.emb)
        with open(BM25_PATH, "w", encoding="utf-8") as f:
            json.dump({"token_lists": self.token_lists}, f, ensure_ascii=False)
        with open(META_PATH, "w", encoding="utf-8") as f:
            json.dump(self.meta, f, ensure_ascii=False)

    def load(self):
        with open(CHUNKS_PATH, encoding="utf-8") as f:
            self.chunks = [json.loads(x) for x in f if x.strip()]
        self.emb = np.load(EMB_PATH)
        with open(BM25_PATH, encoding="utf-8") as f:
            self.token_lists = json.load(f)["token_lists"]
        self.bm25.fit_tokens(self.token_lists)
        if os.path.exists(META_PATH):
            self.meta = json.load(open(META_PATH, encoding="utf-8"))
        return self

    def exists(self):
        return os.path.exists(CHUNKS_PATH) and os.path.exists(EMB_PATH)

    # ---------- 检索 ----------
    def search_dense(self, query, k=20):
        if self.emb is None or len(self.chunks) == 0:
            return []
        q = np.asarray(llm.embed([query])[0], dtype=np.float32)
        n = np.linalg.norm(q)
        if n:
            q /= n
        sims = self.emb @ q
        idx = np.argsort(-sims)[:k]
        return [(int(i), float(sims[i])) for i in idx]

    def search_bm25(self, query, k=20):
        return self.bm25.search(query, k=k)

    def get(self, i):
        return self.chunks[i]


_CACHE = {"kb": None}


def get_kb():
    if _CACHE["kb"] is None:
        kb = KB()
        kb.load()
        _CACHE["kb"] = kb
    return _CACHE["kb"]
