from functools import lru_cache

from sentence_transformers import CrossEncoder, SentenceTransformer

from backend.app.core.config import get_settings


settings = get_settings()


class EmbeddingService:
    """负责把中文文本转换成 BGE-M3 向量。"""

    def __init__(self) -> None:
        # 加载本地 BGE-M3 模型，local_files_only=True 表示不联网下载。
        self.model = SentenceTransformer(settings.embedding_model_path, device=settings.model_device, local_files_only=True)

    def encode(self, texts: list[str]) -> list[list[float]]:
        # 如果没有文本，直接返回空列表。
        if not texts:
            return []
        # normalize_embeddings=True 让向量适合用 COSINE 相似度检索。
        vectors = self.model.encode(
            texts,
            batch_size=settings.embedding_batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        # Milvus 需要普通 Python list，所以把 numpy 数组转成 list。
        return vectors.tolist()




@lru_cache
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService()
