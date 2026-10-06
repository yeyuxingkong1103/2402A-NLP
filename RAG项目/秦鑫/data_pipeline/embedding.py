from __future__ import annotations

import time
from collections.abc import Iterable


def embedding_text(row: dict) -> str:
    return str(row.get("embedding_text") or row.get("text") or row.get("content") or row.get("summary") or row.get("case_summary") or "")


def embedding_batches(rows: list[dict], batch_size: int, max_characters: int | None = None) -> Iterable[list[dict]]:
    batch: list[dict] = []
    characters = 0
    for row in rows:
        length = len(embedding_text(row))
        if batch and (len(batch) >= batch_size or bool(max_characters and characters + length > max_characters)):
            yield batch
            batch, characters = [], 0
        batch.append(row)
        characters += length
    if batch:
        yield batch


def embed(rows: list[dict], model, *, batch_size: int = 32, max_characters: int | None = None, max_batch_chars: int | None = None, expected_dimension: int | None = None, max_retries: int = 1, retry_delay: float = 0.0) -> list[dict]:
    if model is None:
        return [dict(row) for row in rows]
    if max_characters is not None and max_batch_chars is not None and max_characters != max_batch_chars:
        raise ValueError("max_characters 与 max_batch_chars 不能同时指定不同值")
    if batch_size <= 0 or max_retries <= 0:
        raise ValueError("batch_size 和 max_retries 必须大于 0")
    character_limit = max_characters if max_characters is not None else max_batch_chars
    output: list[dict] = []
    for batch in embedding_batches(rows, batch_size, character_limit):
        vectors = None
        for attempt in range(max_retries):
            try:
                vectors = model.embed([embedding_text(row) for row in batch])
                break
            except Exception:
                if attempt + 1 == max_retries:
                    raise
                if retry_delay > 0:
                    time.sleep(retry_delay)
        if vectors is None or len(vectors) != len(batch):
            raise ValueError("向量数量与记录数量不一致")
        for row, vector in zip(batch, vectors, strict=True):
            if expected_dimension is not None and len(vector) != expected_dimension:
                raise ValueError(f"向量维度错误：应为 {expected_dimension}，实际为 {len(vector)}")
            output.append({**dict(row), "embedding": vector})
    return output


def embed_chunks(rows: list[dict], model, **kwargs) -> list[dict]:
    return embed(rows, model, **kwargs)


def _estimate_tokens(value: str) -> int:
    return max(1, len(value) // 4) if value else 0


def _truncate_tokens(value: str, limit: int) -> str:
    text = str(value or "")
    if limit <= 0:
        return ""
    return text if _estimate_tokens(text) <= limit else text[:limit * 4]


def _truncate_utf8(value: str, max_bytes: int) -> str:
    encoded = str(value or "").encode("utf-8")
    return str(value or "") if len(encoded) <= max_bytes else encoded[:max_bytes].decode("utf-8", errors="ignore")


def make_public_batches(collection: str, rows: list[dict], max_records: int = 32, max_characters: int | None = None):
    del collection
    yield from embedding_batches(rows, max_records, max_characters)


def embed_public_batch(collection: str, rows: list[dict], model, expected_dimension: int | None = None, retry_delay: float = 0.0) -> list[dict]:
    del collection
    prepared: list[dict] = []
    for source in rows:
        row = dict(source)
        text = _truncate_tokens(embedding_text(row), 1500)
        row["embedding_text"] = text
        if "content" in row:
            row["content"] = _truncate_utf8(str(row.get("content") or text), 65000)
        prepared.append(row)
    embedded = embed(prepared, model, batch_size=max(1, len(prepared)), expected_dimension=expected_dimension, max_retries=2, retry_delay=retry_delay)
    return [{key: value for key, value in row.items() if key != "embedding_text"} for row in embedded]


__all__ = ["embed", "embed_chunks", "embed_public_batch", "embedding_batches", "embedding_text", "make_public_batches"]
