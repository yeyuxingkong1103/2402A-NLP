"""无需外部模型的本地文本嵌入。"""

from collections import Counter
import hashlib
import math
import re
from typing import Sequence

_TOKEN_RE = re.compile(r"[一-鿿]|[A-Za-z0-9_]+")


class LocalEmbedding:
    """用稳定 token 哈希生成计数向量，并提供余弦相似度。"""

    def __init__(self, dimensions: int = 256) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self.dimensions = dimensions

    def tokenize(self, text: str) -> list[str]:
        return _TOKEN_RE.findall(text.lower())

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token, count in Counter(self.tokenize(text)).items():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:8], "big") % self.dimensions
            vector[index] += float(count)
        return vector

    @staticmethod
    def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
        if len(left) != len(right):
            raise ValueError("vectors must have the same dimensions")
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        if not left_norm or not right_norm:
            return 0.0
        return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)

    similarity = cosine_similarity
