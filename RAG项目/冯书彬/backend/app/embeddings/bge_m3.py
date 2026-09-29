import logging
from collections.abc import Sequence

from sentence_transformers import SentenceTransformer

from backend.app.core.config import settings
from backend.app.embeddings.base import EmbeddingClient

logger = logging.getLogger(__name__)


class BgeM3EmbeddingClient(EmbeddingClient):
    def __init__(self, model_path: str = settings.BGE_M3_MODEL_PATH, model_dtype: str = settings.MODEL_DTYPE) -> None:
        # 保存本地模型路径，初始化时立即按要求加载到 CUDA。
        self.model_path = model_path
        # 保存精度配置，便于 readiness 测试确认使用 FP16。
        self.model_dtype = model_dtype
        # 记录脱敏加载事件，不输出任何用户文本。
        logger.info("loading bge m3 embedding model", extra={"device": "cuda", "dtype": model_dtype})
        # 仅从本地目录加载模型，避免运行时访问远程仓库。
        self.model = SentenceTransformer(model_path, device="cuda", model_kwargs={"dtype": model_dtype}, local_files_only=True)
        # FP16 是任务硬约束；不做 CPU 或其他精度降级。
        self.model.half()
        # 加载完成只记录模型类型和设备，不记录路径细节。
        logger.info("bge m3 embedding model loaded", extra={"device": "cuda"})

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        # 空输入直接返回空列表，避免无意义的模型调用。
        if not texts:
            return []
        # 只记录数量和长度范围，避免日志包含用户文本全文。
        _log_text_batch("embedding input received", texts)
        # SentenceTransformer 负责批量编码；归一化向量便于后续相似度检索。
        embeddings = self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        # 转为 Python list，稳定满足接口约定。
        vectors = embeddings.tolist()
        # 只记录向量数量和维度，不输出向量内容。
        logger.info("embedding output produced", extra={"count": len(vectors), "dimension": len(vectors[0]) if vectors else 0})
        return vectors


def _log_text_batch(message: str, texts: Sequence[str]) -> None:
    # 统计文本长度用于排查异常输入，同时避免泄露具体文本。
    lengths = [len(text) for text in texts]
    # 记录批大小和长度边界，不记录原文。
    logger.info(message, extra={"count": len(texts), "min_chars": min(lengths), "max_chars": max(lengths)})
