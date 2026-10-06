from __future__ import annotations

"""本地 Embedding 模型封装，将问题和文档文本转换为向量。"""

from typing import Iterable

from ..config.config import EmbeddingConfig


class LocalEmbeddingClient:
    """封装 sentence-transformers 本地向量模型。"""
    def __init__(self, config: EmbeddingConfig) -> None:
        self.config = config
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:  # pragma: no cover - 依赖由运行环境提供
            raise RuntimeError(
                "缺少 sentence-transformers，请先运行：pip install sentence-transformers"
            ) from error

        self.model = SentenceTransformer(config.model_name, device=config.device)
        self.model.max_seq_length = config.max_seq_length

    def encode(self, texts: Iterable[str]) -> list[list[float]]:
        """批量编码文本并返回 Python 浮点数组。"""
        text_list = list(texts)
        if not text_list:
            return []

        embeddings = self.model.encode(
            text_list,
            batch_size=self.config.batch_size,
            normalize_embeddings=self.config.normalize_embeddings,
            show_progress_bar=True,
        )
        return embeddings.tolist()


def build_embedding_text(content: str, metadata: dict[str, str]) -> str:
    """把标题和分类元数据拼到正文前，形成向量模型输入。"""
    fields = [
        metadata.get("title", ""),
        metadata.get("subject", ""),
        metadata.get("grade", ""),
        metadata.get("textbook_version", ""),
        metadata.get("document_type", ""),
        content,
    ]
    return "\n".join(field for field in fields if field)
