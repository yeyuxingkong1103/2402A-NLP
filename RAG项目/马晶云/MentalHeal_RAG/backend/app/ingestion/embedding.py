from pathlib import Path
from typing import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer


# 这个类专门负责“把文本变成向量”。
# 入向量库的时候，每个 chunk 的正文都会先经过这里编码。
class BgeEmbedder:
    def __init__(self, model_path: str | Path, batch_size: int = 16) -> None:
        # 加载 embedding 模型。这个模型的作用不是生成回答，而是把文本变成一串数字向量。
        self.model = SentenceTransformer(str(model_path))
        # batch_size 表示一次处理多少段文本；太小会慢，太大可能占内存。
        self.batch_size = batch_size

    @property
    def dimension(self) -> int:
        # 返回向量维度，比如 BGE-m3 常见是 1024 维。
        # Milvus 建 collection 的时候必须知道这个维度，否则向量存不进去。
        return int(self.model.get_sentence_embedding_dimension())

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        # 如果没有文本，就直接返回空列表，避免后面模型空跑。
        if not texts:
            return []

        # 真正的编码步骤：把一批文本转成一批向量。
        vectors = self.model.encode(
            # SentenceTransformer 更习惯接收 list，所以这里转一下。
            list(texts),
            # 按 batch_size 分批编码，避免一次性塞太多文本。
            batch_size=self.batch_size,
            # 归一化后更适合用余弦相似度做检索。
            normalize_embeddings=True,
            # 先转成 numpy，方便后面统一数据类型。
            convert_to_numpy=True,
            # 命令行或后端服务里不显示进度条，输出更干净。
            show_progress_bar=False,
        )
        # Milvus 最终需要普通 Python list，而且向量一般用 float32 就够了。
        return np.asarray(vectors, dtype=np.float32).tolist()
