from __future__ import annotations

import hashlib
from collections import OrderedDict
from threading import RLock
from time import monotonic

import httpx

from ..config import Settings
from .llm import require_siliconflow_key


class EmbeddingCache:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._items: OrderedDict[str, tuple[float, list[float]]] = OrderedDict()
        self._lock = RLock()

    def key(self, text: str) -> str:
        digest = hashlib.sha256(str(text or "").encode("utf-8", "ignore")).hexdigest()
        return f"{self.settings.embedding_model}:{digest}"

    def _config(self) -> tuple[int, int]:
        return (int(getattr(self.settings, "embedding_cache_size", 2048) or 2048), int(getattr(self.settings, "embedding_cache_ttl", 86400) or 86400))

    def get(self, text: str) -> list[float] | None:
        with self._lock:
            key = self.key(text)
            item = self._items.get(key)
            if item is None:
                return None
            created, vector = item
            if monotonic() - created > self._config()[1]:
                self._items.pop(key, None)
                return None
            self._items.move_to_end(key)
            return list(vector)

    def set(self, text: str, vector: list[float]) -> None:
        with self._lock:
            self._items[self.key(text)] = (monotonic(), list(vector))
            self._items.move_to_end(self.key(text))
            while len(self._items) > self._config()[0]:
                self._items.popitem(last=False)


class EmbeddingClient:
    def __init__(self, settings: Settings, cache: EmbeddingCache | None = None):
        self.settings = settings
        self.cache = cache or EmbeddingCache(settings)
        self.last_error = ""

    def embed(self, texts: list[str]) -> list[list[float]]:
        normalized = [str(text or "") for text in texts]
        if not normalized:
            return []
        ready, missing = {}, []
        for text in dict.fromkeys(normalized):
            vector = self.cache.get(text)
            if vector is None:
                missing.append(text)
            else:
                ready[text] = vector
        if missing:
            try:
                require_siliconflow_key(self.settings)
                response = httpx.post(f"{self.settings.siliconflow_base_url.rstrip('/')}/embeddings", headers={"Authorization": f"Bearer {self.settings.siliconflow_api_key}"}, json={"model": self.settings.embedding_model, "input": missing}, timeout=getattr(self.settings, "embedding_timeout", 60))
                response.raise_for_status()
                vectors = [row["embedding"] for row in response.json()["data"]]
                if len(vectors) != len(missing):
                    raise ValueError("Embedding 返回数量与输入文本数量不一致")
                for text, vector in zip(missing, vectors, strict=True):
                    ready[text] = list(vector)
                    self.cache.set(text, vector)
                self.last_error = ""
            except Exception as exc:
                self.last_error = str(exc)[:300]
                raise
        return [list(ready[text]) for text in normalized]
