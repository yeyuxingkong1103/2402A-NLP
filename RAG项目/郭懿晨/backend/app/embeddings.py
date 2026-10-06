from collections import Counter
from hashlib import md5
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel

try:
    import torch
except ImportError:
    torch = None
else:
    out_of_memory_error = getattr(torch.cuda, "OutOfMemoryError", None)
    if out_of_memory_error is not None:
        torch.OutofMemoryError = out_of_memory_error

try:
    from FlagEmbedding import BGEM3FlagModel
except ImportError:
    BGEM3FlagModel = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


class EmbeddingResult(BaseModel):
    """bge-m3 嵌入结果。"""

    dense: list[float]
    sparse_indices: list[int]
    sparse_values: list[float]

    def to_qdrant_vectors(self) -> dict[str, object]:
        """转换为 Qdrant 命名向量格式。"""
        return {
            "dense": self.dense,
            "sparse": {"indices": self.sparse_indices, "values": self.sparse_values},
        }


class TextEmbedder(Protocol):
    """文本嵌入器协议。"""

    def embed_texts(self, texts: list[str]) -> list[EmbeddingResult]:
        """生成文本嵌入。"""


class BgeM3Embedder:
    """本地 bge-m3 嵌入器。"""

    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path
        if not self.model_path.exists():
            raise FileNotFoundError(f"bge-m3 模型路径不存在: {self.model_path}")

        self.mode = "sentence-transformers"
        self.model = None

        if BGEM3FlagModel is not None and self._can_use_flagembedding_model(self.model_path):
            try:
                self.model = BGEM3FlagModel(str(self.model_path), use_fp16=False)
                self.mode = "flagembedding"
            except Exception:
                self.model = None
                self.mode = "sentence-transformers"

        if self.model is None:
            if SentenceTransformer is None:
                raise ImportError("无法加载 bge-m3：FlagEmbedding 和 sentence-transformers 都不可用")
            self.model = SentenceTransformer(str(self.model_path))

    def _can_use_flagembedding_model(self, model_path: Path) -> bool:
        """只有模型目录完整时才启用 FlagEmbedding。"""
        return (model_path / "config.json").exists() and (
            (model_path / "model.safetensors").exists() or (model_path / "pytorch_model.bin").exists()
        )

    def embed_texts(self, texts: list[str]) -> list[EmbeddingResult]:
        """生成 dense 与 sparse 表示。"""
        if not texts:
            return []

        if self.mode == "flagembedding":
            outputs = self.model.encode(texts, return_dense=True, return_sparse=True)
            dense_vectors = outputs["dense_vecs"]
            sparse_vectors = outputs["lexical_weights"]
            results: list[EmbeddingResult] = []
            for dense, sparse in zip(dense_vectors, sparse_vectors, strict=True):
                indices = [int(index) for index in sparse.keys()]
                values = [float(value) for value in sparse.values()]
                results.append(
                    EmbeddingResult(
                        dense=[float(value) for value in dense],
                        sparse_indices=indices,
                        sparse_values=values,
                    )
                )
            return results

        dense_vectors = self.model.encode(texts)
        results: list[EmbeddingResult] = []
        for text, dense in zip(texts, dense_vectors, strict=True):
            sparse_indices, sparse_values = self._build_sparse_vector(text)
            results.append(
                EmbeddingResult(
                    dense=[float(value) for value in dense],
                    sparse_indices=sparse_indices,
                    sparse_values=sparse_values,
                )
            )
        return results

    def _build_sparse_vector(self, text: str) -> tuple[list[int], list[float]]:
        """用稳定的字符哈希生成简单稀疏向量。"""
        tokens = [character for character in text if not character.isspace()]
        counts = Counter(tokens)
        indices: list[int] = []
        values: list[float] = []
        used_indices: set[int] = set()
        for token, count in sorted(counts.items(), key=lambda item: item[0]):
            token_hash = int(md5(token.encode("utf-8")).hexdigest()[:8], 16) % 100_000
            while token_hash in used_indices:
                token_hash = (token_hash + 1) % 100_000
            used_indices.add(token_hash)
            indices.append(token_hash)
            values.append(float(count))
        return indices, values


class FakeBgeM3Embedder:
    """测试用嵌入器。"""

    def __init__(self, size: int = 3) -> None:
        self.size = size

    def embed_texts(self, texts: list[str]) -> list[EmbeddingResult]:
        """生成稳定测试向量。"""
        return [
            EmbeddingResult(
                dense=[1.0] + [0.0] * (self.size - 1),
                sparse_indices=[0],
                sparse_values=[1.0],
            )
            for _ in texts
        ]
