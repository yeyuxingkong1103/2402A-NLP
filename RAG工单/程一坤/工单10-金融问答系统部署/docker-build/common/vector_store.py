# -*- coding: utf-8 -*-
"""
本地向量库
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
说明：轻量级向量存储（numpy + JSON 持久化），余弦相似度检索。
     数据规模（两份招股说明书约千级分块）下性能足够，且零外部依赖。
"""
import os
import json
import numpy as np

from config import INDEX_DIR
from ollama_client import client


class VectorStore:
    """基于 numpy 的本地向量库"""

    def __init__(self):
        self.texts = []      # 分块文本
        self.metadatas = []  # 元数据 [{"page":.., "source":..}, ...]
        self.vecs = None     # np.ndarray [n, dim]

    # ── 构建 ───────────────────────────────────────────────
    def add(self, chunks, source_name):
        """向量化并写入
        参数 chunks: [{"text": str, "page": int}, ...]
        """
        print(f"[VectorStore] 向量化 {len(chunks)} 个分块 (来源: {source_name}) ...")
        vecs = client.embed_batch([c["text"] for c in chunks])
        for c, v in zip(chunks, vecs):
            self.texts.append(c["text"])
            self.metadatas.append({"page": c.get("page"), "source": source_name})
        new = np.array(vecs, dtype=np.float32)
        self.vecs = new if self.vecs is None else np.vstack([self.vecs, new])
        print(f"[VectorStore] 完成，当前总量: {len(self.texts)}")

    # ── 检索 ───────────────────────────────────────────────
    @staticmethod
    def _cosine(a, b):
        """余弦相似度: a [n,d] 与 b [d]"""
        na, nb = np.linalg.norm(a, axis=1), np.linalg.norm(b)
        return (a @ b) / (na * nb + 1e-8)

    def search(self, query, top_k=5):
        """查询向量化后做余弦相似度检索
        返回: [{"text", "page", "source", "score"}, ...]
        """
        if self.vecs is None or len(self.texts) == 0:
            return []
        qv = np.array(client.embed(query), dtype=np.float32)
        scores = self._cosine(self.vecs, qv)
        idx = np.argsort(-scores)[:top_k]
        return [
            {
                "text": self.texts[i],
                "page": self.metadatas[i].get("page"),
                "source": self.metadatas[i].get("source"),
                "score": float(scores[i]),
            }
            for i in idx
        ]

    # ── 持久化 ─────────────────────────────────────────────
    def save(self, name):
        os.makedirs(INDEX_DIR, exist_ok=True)
        data = {
            "texts": self.texts,
            "metadatas": self.metadatas,
            "vecs": self.vecs.tolist() if self.vecs is not None else None,
        }
        path = os.path.join(INDEX_DIR, f"{name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        print(f"[VectorStore] 已保存: {path}")

    @classmethod
    def load(cls, name):
        path = os.path.join(INDEX_DIR, f"{name}.json")
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        store = cls()
        store.texts = data["texts"]
        store.metadatas = data["metadatas"]
        store.vecs = np.array(data["vecs"], dtype=np.float32) if data["vecs"] else None
        print(f"[VectorStore] 已加载 {len(store.texts)} 条: {path}")
        return store
