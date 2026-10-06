# 工单编号：人工智能NLP-RAG-图像内容解析及检索优化
"""向量化与检索：BGE-M3 混合召回（RRF 融合）+ 重排序精排 + 持久化

工单2 的检索优化点：
  1. 召回与精排分离——粗排放宽到 RECALL_K 个保证不漏，精排收紧到 TOP_K 个保证信噪比
  2. 新增 bge-reranker-v2-m3 重排序，用交叉编码器逐对打分，比向量相似度准得多
"""
import json
import threading
from pathlib import Path

import numpy as np

from config import (
    BGE_M3_PATH, RERANKER_PATH, RERANK_MAX_CHARS, USE_FP16, TOP_K, RRF_K,
    DENSE_WEIGHT, SPARSE_WEIGHT,
)

# 显存只有 6GB，而 BGE-M3 和重排序模型同时常驻。并发请求同时推理会互相抢显存，
# 所以把 GPU 前向串行化——接口仍可并发，只是真正落到 GPU 的那一步排队执行。
_GPU_LOCK = threading.Lock()


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
        with _GPU_LOCK:
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


# ---------------- 重排序（工单2 新增的检索优化） ----------------
class BGEReranker:
    """bge-reranker-v2-m3 交叉编码器重排序。

    向量检索是双塔结构：query 和 chunk 各自独立编码后才比相似度，精度有限。
    重排序把两者拼成一对直接过模型，能看出词序和上下文关系，排序质量明显更好，
    代价是每个候选都要前向一次，所以只对少量召回结果做。
    """

    def __init__(self, model_path=RERANKER_PATH, use_fp16=USE_FP16):
        from FlagEmbedding import FlagReranker
        self.model = FlagReranker(model_path, use_fp16=use_fp16)

    def rerank(self, query, candidates, top_k=TOP_K, max_chars=RERANK_MAX_CHARS):
        """对候选逐对打分后取前 top_k。

        max_chars 是送给重排序模型的正文长度上限。重排序模型本身只吃 512 个
        token，喂更长也会被截断，反而白白拉长前向时间 —— 实测候选 20 个时，
        喂全长要 1.2 秒，截断后明显更快。
        """
        if not candidates:
            return []
        pairs = [(query, chunk["text"][:max_chars]) for chunk, _ in candidates]
        with _GPU_LOCK:
            scores = self.model.compute_score(pairs)
        if isinstance(scores, (int, float)):        # 只有一对时返回的是标量
            scores = [scores]
        ranked = sorted(zip(candidates, scores), key=lambda item: -item[1])
        return [(chunk, float(score)) for (chunk, _), score in ranked[:top_k]]


_reranker = None


def rerank(query, candidates, top_k=TOP_K, max_chars=RERANK_MAX_CHARS):
    """模块级入口，重排序模型全局只加载一次。"""
    global _reranker
    if _reranker is None:
        _reranker = BGEReranker()
    return _reranker.rerank(query, candidates, top_k, max_chars)
