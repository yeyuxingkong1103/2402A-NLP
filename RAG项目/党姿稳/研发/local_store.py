"""
local_store.py — 向量库的本地降级实现

向量常驻内存做检索，数据以 JSON 落盘，进程重启后自动恢复。
目的是让项目在没有 Milvus 的机器上（本地调试、单元测试）也能跑通全链路。

数据规模建议控制在万条以内；生产环境请使用 vector_store.MilvusStore。
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import config
from vector_store import VectorStore


class LocalStore(VectorStore):
    """numpy 内存索引 + JSON 持久化。"""

    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path else config.STORAGE_DIR / "vector_store.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._data: dict[str, list[dict]] = {}
        self._next_id = 1
        self._load()

    # ------------------------------------------------------------ 持久化

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            self._data = payload.get("collections", {})
            self._next_id = int(payload.get("next_id", 1))
        except (json.JSONDecodeError, OSError, ValueError):
            # 文件损坏时从空库开始，避免整个服务起不来
            self._data = {}
            self._next_id = 1

    def _save(self) -> None:
        payload = {"collections": self._data, "next_id": self._next_id}
        temp = self.path.with_suffix(".tmp")
        temp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        temp.replace(self.path)  # 原子替换，防止写一半崩溃损坏文件

    # ------------------------------------------------------------ 工具

    @staticmethod
    def _match(record: dict, filters: dict[str, Any] | None) -> bool:
        if not filters:
            return True
        return all(record.get(key) == value for key, value in filters.items())

    @staticmethod
    def _strip_embedding(record: dict) -> dict:
        return {k: v for k, v in record.items() if k != "embedding"}

    # ------------------------------------------------------------ 接口实现

    def upsert(self, collection: str, records: list[dict]) -> list[int]:
        if not records:
            return []

        with self._lock:
            bucket = self._data.setdefault(collection, [])
            now = int(time.time())
            written: list[int] = []

            for record in records:
                row = dict(record)
                row["id"] = self._next_id
                self._next_id += 1
                row.setdefault("created_at", now)
                row.setdefault("updated_at", now)

                # 同 source + 同 text 视为同一条，覆盖旧记录以实现幂等导入
                replaced = False
                for idx, existing in enumerate(bucket):
                    if (
                        existing.get("text") == row.get("text")
                        and existing.get("source") == row.get("source")
                    ):
                        row["id"] = existing["id"]
                        row["created_at"] = existing.get("created_at", now)
                        bucket[idx] = row
                        replaced = True
                        break
                if not replaced:
                    bucket.append(row)
                written.append(int(row["id"]))

            self._save()
            return written

    def search(self, collection, vector, top_k=10, filters=None) -> list[dict]:
        import numpy as np

        with self._lock:
            bucket = [r for r in self._data.get(collection, []) if self._match(r, filters)]
            matrix = np.asarray([r.get("embedding") or [] for r in bucket], dtype=np.float32)

        if not bucket:
            return []

        query = np.asarray(vector, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[1] != query.shape[0]:
            return []

        scores = matrix @ query  # 向量均已归一化，点积即余弦相似度
        order = np.argsort(-scores)[:top_k]

        hits: list[dict] = []
        for idx in order:
            hit = self._strip_embedding(bucket[int(idx)])
            hit["score"] = float(scores[int(idx)])
            hits.append(hit)
        return hits

    def query(self, collection, filters=None, limit=100, offset=0) -> list[dict]:
        with self._lock:
            bucket = [r for r in self._data.get(collection, []) if self._match(r, filters)]

        bucket.sort(key=lambda r: (r.get("created_at", 0), r.get("id", 0)), reverse=True)
        return [self._strip_embedding(r) for r in bucket[offset : offset + limit]]

    def delete(self, collection, filters) -> int:
        with self._lock:
            bucket = self._data.get(collection, [])
            kept = [r for r in bucket if not self._match(r, filters)]
            removed = len(bucket) - len(kept)
            self._data[collection] = kept
            if removed:
                self._save()
            return removed

    def count(self, collection) -> int:
        with self._lock:
            return len(self._data.get(collection, []))

    def drop(self, collection) -> None:
        with self._lock:
            if collection in self._data:
                del self._data[collection]
                self._save()

    def list_collections(self) -> list[str]:
        with self._lock:
            return list(self._data.keys())
