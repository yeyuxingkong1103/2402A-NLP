# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
"""向量化与检索：本地 BGE-M3（Dense + Sparse 混合检索，RRF 融合）+ 持久化"""
import json
from pathlib import Path

import numpy as np

from config import (
    BGE_M3_PATH, USE_FP16, TOP_K, RRF_K, DENSE_WEIGHT, SPARSE_WEIGHT,
)


class BGEM3VectorStore:
    """BGE-M3 同时给出稠密(语义)和稀疏(词权重)两种向量。

    两路分数尺度完全不同，直接加权求和会让稀疏分靠一次偶然的词汇重叠就顶掉
    语义分，所以用 RRF（按名次融合，与分数尺度无关）而不是分数相加。
    """

    def __init__(self, model_path=BGE_M3_PATH, use_fp16=USE_FP16):
        from FlagEmbedding import BGEM3FlagModel
        self.model = BGEM3FlagModel(model_path, use_fp16=use_fp16)
        self.chunks = []
        self.dense_emb = None
        self.sparse_emb = None

    # ---------- 编码 ----------
    def _encode(self, texts, batch_size=16, max_length=1024):
        out = self.model.encode(
            texts, batch_size=batch_size, max_length=max_length,
            return_dense=True, return_sparse=True, return_colbert_vecs=False,
        )
        dense = np.asarray(out["dense_vecs"], dtype=np.float32)
        dense /= (np.linalg.norm(dense, axis=1, keepdims=True) + 1e-12)
        return dense, out["lexical_weights"]

    # ---------- 建索引 ----------
    def build(self, chunks, batch_size=16):
        self.chunks = chunks
        print(f"[VectorStore] 编码 {len(chunks)} 个块 ...")
        self.dense_emb, self.sparse_emb = self._encode(
            [c["text"] for c in chunks], batch_size=batch_size
        )
        print(f"[VectorStore] 索引完成 dense={self.dense_emb.shape}")

    # ---------- 持久化（知识库管理） ----------
    def save(self, out_dir):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "dense.npy", self.dense_emb)
        (out_dir / "chunks.json").write_text(
            json.dumps(self.chunks, ensure_ascii=False), encoding="utf-8"
        )
        # 稀疏权重是 float16，JSON 序列化不了，default=float 转成 Python float。
        # 不转的话写一半就抛 TypeError，只会留下一个截断的坏文件。
        (out_dir / "sparse.json").write_text(
            json.dumps(self.sparse_emb, ensure_ascii=False, default=float),
            encoding="utf-8",
        )

    def load(self, in_dir):
        """加载缓存。文件缺失或损坏都返回 False，由调用方重新构建（容错）。"""
        in_dir = Path(in_dir)
        if not (in_dir / "dense.npy").exists() or not (in_dir / "chunks.json").exists():
            return False
        try:
            dense = np.load(in_dir / "dense.npy")
            chunks = json.loads((in_dir / "chunks.json").read_text(encoding="utf-8"))
            sparse_file = in_dir / "sparse.json"
            sparse = (json.loads(sparse_file.read_text(encoding="utf-8"))
                      if sparse_file.exists() else None)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"[VectorStore] 缓存损坏，将重新构建：{exc}")
            return False

        if len(chunks) != len(dense):          # 索引与分块对不上，同样视为损坏
            print("[VectorStore] 缓存与索引不一致，将重新构建")
            return False

        self.dense_emb, self.chunks, self.sparse_emb = dense, chunks, sparse
        return True

    # ---------- 检索 ----------
    @staticmethod
    def _sparse_score(query_weights, doc_weights):
        if not query_weights or not doc_weights:
            return 0.0
        return float(sum(query_weights.get(t, 0.0) * doc_weights.get(t, 0.0)
                         for t in query_weights))

    @staticmethod
    def _ranks(scores):
        """把分数转成名次(0 最好)，用于 RRF。"""
        order = np.argsort(-scores)
        ranks = np.empty(len(scores), dtype=np.int32)
        ranks[order] = np.arange(len(scores))
        return ranks

    def search(self, query, top_k=TOP_K):
        if self.dense_emb is None or not self.chunks:
            return []

        query_dense, query_sparse = self._encode([query])
        dense_scores = self.dense_emb @ query_dense[0]

        if query_sparse and self.sparse_emb:
            sparse_scores = np.array(
                [self._sparse_score(query_sparse[0], d) for d in self.sparse_emb],
                dtype=np.float32,
            )
        else:
            sparse_scores = None

        fused = DENSE_WEIGHT / (RRF_K + self._ranks(dense_scores))
        if sparse_scores is not None and sparse_scores.max() > 0:
            fused += SPARSE_WEIGHT / (RRF_K + self._ranks(sparse_scores))

        idx = np.argsort(-fused)[:top_k]
        return [(self.chunks[int(i)], float(fused[int(i)])) for i in idx]
