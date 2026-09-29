"""
embeddings.py — 文本向量化封装

生产环境使用 BGE-m3（1024 维稠密向量）。
本地测试若不想下载数 GB 的模型，可把 config.EMBEDDING_BACKEND 设为 "hash"，
使用基于字符 n-gram 的哈希向量作为降级实现。

注意：hash 后端只捕捉字面重叠，没有真正的语义理解能力，
仅用于打通链路，生产环境务必使用 bge-m3。
"""

from __future__ import annotations

import hashlib
import re
import threading
from typing import Sequence

import numpy as np

import config

# 中英文混排的粗粒度切分：英文单词、数字、单个汉字
_TOKEN_RE = re.compile(r"[a-zA-Z]+|[0-9]+|[一-鿿]")

_model_cache: dict[str, object] = {}

# 知识库检索与长期记忆检索是并行的，两个线程可能同时首次访问 model。
# 不加锁会各自加载一份 2.3GB 权重，内存直接翻倍。
_bge_lock = threading.Lock()


# ---------------------------------------------------------------- 工具函数


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """按行做 L2 归一化，零向量保持为零向量。"""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def _extract_features(text: str) -> list[str]:
    """抽取用于哈希的特征：词语、字、二元组。

    中文按字切分后组合成二元组，能捕获"劳动"+"合同"这类固定搭配；
    英文保留完整单词，避免把单词切碎。
    """
    base = _TOKEN_RE.findall(text.lower())
    features = list(base)

    # 相邻二元组，弥补单词级特征的稀疏性
    features.extend(f"{base[i]}_{base[i + 1]}" for i in range(len(base) - 1))

    # 中文连续片段整体作为一个特征，帮助区分"劳动合同"与"劳动" + "合同"
    for match in re.findall(r"[一-鿿]{2,}", text):
        features.append(match)

    return features


def _feature_index(token: str, dim: int) -> tuple[int, float]:
    """把特征映射到 [0, dim) 的下标，并给出 +1/-1 符号。

    使用 md5 而非内置 hash()，因为后者在不同进程间结果不稳定，
    会导致入库与查询时向量空间不一致。
    """
    digest = hashlib.md5(token.encode("utf-8")).digest()
    index = int.from_bytes(digest[:4], "big") % dim
    sign = 1.0 if digest[4] & 1 else -1.0
    return index, sign


# ---------------------------------------------------------------- 后端实现


class HashEncoder:
    """离线降级向量化：特征哈希 + 词频加权。

    完全没有外部依赖和模型下载，代价是只能做字面匹配。
    """

    def __init__(self, dim: int = config.EMBEDDING_DIM):
        self.dim = dim

    def _encode_one(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=np.float32)
        features = _extract_features(text or "")
        if not features:
            return vector

        # sublinear tf：抑制高频特征的权重，避免长文本被少数词主导
        counts: dict[str, int] = {}
        for token in features:
            counts[token] = counts.get(token, 0) + 1

        for token, count in counts.items():
            index, sign = _feature_index(token, self.dim)
            vector[index] += sign * (1.0 + np.log(count))

        return vector

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        matrix = np.vstack([self._encode_one(t) for t in texts])
        return _l2_normalize(matrix).astype(np.float32).tolist()

    def encode_query(self, query: str) -> list[float]:
        return self.encode([query])[0]


class BGEM3Encoder:
    """BGE-m3 向量化，基于 sentence-transformers 加载。

    模型体积较大（约 2GB），首次使用时才真正加载并常驻内存。
    """

    def __init__(self, model_name: str = config.EMBEDDING_MODEL):
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            with _bge_lock:
                # 拿到锁后要再判一次：等锁期间别的线程可能已经加载好了
                if self._model is None:
                    try:
                        from sentence_transformers import SentenceTransformer
                    except ImportError as exc:  # pragma: no cover - 依赖缺失时的友好提示
                        raise RuntimeError(
                            "使用 BGE-m3 需要安装 sentence-transformers："
                            "pip install sentence-transformers\n"
                            "或把 EMBEDDING_BACKEND 改为 hash 使用离线降级方案。"
                        ) from exc

                    self._model = SentenceTransformer(self.model_name, trust_remote_code=True)
        return self._model

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self.model.encode(
            list(texts),
            batch_size=config.EMBEDDING_BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 64,
        )
        return np.asarray(vectors, dtype=np.float32).tolist()

    def encode_query(self, query: str) -> list[float]:
        # bge-m3 对查询和文档使用同一编码方式，无需额外指令前缀
        return self.encode([query])[0]


# ---------------------------------------------------------------- 对外接口


def get_encoder():
    """按配置返回向量化器（进程内单例）。"""
    backend = config.EMBEDDING_BACKEND
    if backend not in _model_cache:
        aliases = {"bge": "bge-m3", "bge_m3": "bge-m3", "bge-m3": "bge-m3"}
        normalized = aliases.get(backend.lower(), backend.lower())

        if normalized == "bge-m3":
            _model_cache[backend] = BGEM3Encoder()
        else:
            _model_cache[backend] = HashEncoder()
    return _model_cache[backend]


def encode_texts(texts: Sequence[str]) -> list[list[float]]:
    """批量编码为向量。"""
    return get_encoder().encode(texts)


def encode_query(query: str) -> list[float]:
    """编码单条查询。"""
    return get_encoder().encode_query(query)


def dimension() -> int:
    """当前向量维度，用于校验与 Milvus schema 是否匹配。"""
    return config.EMBEDDING_DIM


if __name__ == "__main__":
    # 自检：确认维度正确、归一化生效、相同文本向量一致
    vecs = encode_texts(["劳动合同解除的赔偿标准", "劳动合同解除的赔偿标准", "今天天气不错"])
    assert len(vecs) == 3, "批量编码返回条数不符"
    assert len(vecs[0]) == dimension(), f"向量维度应为 {dimension()}"

    norm = float(np.linalg.norm(vecs[0]))
    assert abs(norm - 1.0) < 1e-5, f"向量未归一化，范数={norm}"

    same = float(np.dot(vecs[0], vecs[1]))
    diff = float(np.dot(vecs[0], vecs[2]))
    print(f"后端      : {config.EMBEDDING_BACKEND}")
    print(f"维度      : {len(vecs[0])}")
    print(f"相同文本  : 余弦={same:.4f}")
    print(f"无关文本  : 余弦={diff:.4f}")
    assert same > diff, "相同文本的相似度应高于无关文本"
    print("embeddings 自检通过。")
