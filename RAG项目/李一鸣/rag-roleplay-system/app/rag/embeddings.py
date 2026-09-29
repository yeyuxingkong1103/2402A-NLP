import hashlib
import logging
import re
from collections.abc import Sequence

import numpy as np

from app.core.config import Settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """BGE-m3 adapter with a deterministic local fallback for development."""

    def __init__(self, settings: Settings):
        self.provider = settings.embedding_provider.lower()
        self.model_name = settings.embedding_model
        self.dimension = settings.embedding_dimension
        self._model = None
        self._fallback_warned = False

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if self.provider in {"bge_m3", "flag_embedding", "sentence_transformers"}:
            vectors = self._try_model(texts)
            if vectors is not None:
                return vectors
        return [self._hash_embedding(text) for text in texts]

    def _try_model(self, texts: Sequence[str]) -> list[list[float]] | None:
        if self._model is None:
            try:
                if self.provider in {"bge_m3", "flag_embedding"}:
                    from FlagEmbedding import BGEM3FlagModel  # type: ignore

                    self._model = BGEM3FlagModel(self.model_name, use_fp16=False)
                else:
                    from sentence_transformers import SentenceTransformer  # type: ignore

                    self._model = SentenceTransformer(self.model_name)
                logger.info("embedding model loaded: %s", self.model_name)
            except Exception:
                if not self._fallback_warned:
                    logger.exception("embedding model unavailable; using hash fallback")
                    self._fallback_warned = True
                return None
        try:
            if self.provider in {"bge_m3", "flag_embedding"}:
                result = self._model.encode(list(texts), batch_size=8, max_length=8192)
                vectors = result["dense_vecs"] if isinstance(result, dict) else result
            else:
                vectors = self._model.encode(list(texts), normalize_embeddings=True)
            return [self._normalize(vector).tolist() for vector in vectors]
        except Exception:
            logger.exception("embedding inference failed; using hash fallback")
            return None

    def _hash_embedding(self, text: str) -> list[float]:
        vector = np.zeros(self.dimension, dtype=np.float32)
        tokens = re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]|[^\s]", text.lower())
        if not tokens:
            return vector.tolist()
        for index, token in enumerate(tokens):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            position = int.from_bytes(digest, "big") % self.dimension
            vector[position] += 1.0 + (index % 5) * 0.05
        return self._normalize(vector).tolist()

    @staticmethod
    def _normalize(vector: Sequence[float]) -> np.ndarray:
        array = np.asarray(vector, dtype=np.float32)
        norm = float(np.linalg.norm(array))
        return array if norm == 0 else array / norm
