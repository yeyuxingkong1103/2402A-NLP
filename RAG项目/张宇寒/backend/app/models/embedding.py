from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from collections.abc import Iterable
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
        self._unavailable_until = 0.0
        self._circuit_lock = RLock()

    def _check_circuit(self) -> None:
        with self._circuit_lock:
            if monotonic() < self._unavailable_until:
                detail = f"：{self.last_error}" if self.last_error else ""
                raise RuntimeError(f"Embedding 服务暂时不可用，请稍后重试{detail}")

    def _open_circuit(self) -> None:
        cooldown = max(0, int(getattr(self.settings, "embedding_failure_cooldown", 30)))
        with self._circuit_lock:
            self._unavailable_until = monotonic() + cooldown

    def _close_circuit(self) -> None:
        with self._circuit_lock:
            self._unavailable_until = 0.0

    def _batches(self, texts: list[str]) -> Iterable[list[str]]:
        batch_size = max(1, int(getattr(self.settings, "embedding_batch_size", 32) or 32))
        max_characters = int(getattr(self.settings, "embedding_batch_max_chars", 0) or 0)
        batch: list[str] = []
        characters = 0
        for text in texts:
            text_length = len(text)
            if batch and (len(batch) >= batch_size or (max_characters and characters + text_length > max_characters)):
                yield batch
                batch = []
                characters = 0
            batch.append(text)
            characters += text_length
        if batch:
            yield batch

    def embed(self, texts: list[str]) -> list[list[float]]:
        normalized = [str(text or "") for text in texts]
        if not normalized:
            return []
        ready: dict[str, list[float]] = {}
        unique_texts = list(dict.fromkeys(normalized))
        try:
            require_siliconflow_key(self.settings)
            self._check_circuit()
            retries = max(0, int(getattr(self.settings, "embedding_retry_count", 1)))
            delay = max(0.0, float(getattr(self.settings, "embedding_retry_delay", 1.0)))
            for batch in self._batches(unique_texts):
                missing: list[str] = []
                for text in batch:
                    vector = self.cache.get(text)
                    if vector is None:
                        missing.append(text)
                    else:
                        ready[text] = vector
                if not missing:
                    continue
                for attempt in range(retries + 1):
                    try:
                        response = httpx.post(
                            f"{self.settings.siliconflow_embedding_base_url.rstrip('/')}/embeddings",
                            headers={"Authorization": f"Bearer {self.settings.siliconflow_api_key}"},
                            json={"model": self.settings.embedding_model, "input": missing},
                            timeout=httpx.Timeout(
                                getattr(self.settings, "embedding_timeout", 60),
                                connect=getattr(self.settings, "embedding_connect_timeout", 5),
                            ),
                        )
                        response.raise_for_status()
                        vectors = [row["embedding"] for row in response.json()["data"]]
                        if len(vectors) != len(missing):
                            raise ValueError("Embedding 返回数量与输入文本数量不一致")
                        for text, vector in zip(missing, vectors, strict=True):
                            ready[text] = list(vector)
                            self.cache.set(text, vector)
                        break
                    except httpx.HTTPError as exc:
                        response = getattr(exc, "response", None)
                        status_code = response.status_code if response is not None else None
                        retryable = (
                            isinstance(exc, (httpx.TimeoutException, httpx.ConnectError))
                            or status_code == 429
                            or (status_code is not None and status_code >= 500)
                        )
                        if not retryable:
                            raise
                        if attempt >= retries:
                            self._open_circuit()
                            raise
                        retry_after = 0.0
                        if status_code == 429:
                            try:
                                retry_after = float(response.headers.get("retry-after", "0"))
                            except (TypeError, ValueError):
                                retry_after = 0.0
                            retry_after = max(retry_after, 15.0)
                        time.sleep(max(delay * (2 ** attempt), retry_after))
            self.last_error = ""
            self._close_circuit()
        except Exception as exc:
            self.last_error = str(exc)[:300]
            raise
        return [list(ready[text]) for text in normalized]
