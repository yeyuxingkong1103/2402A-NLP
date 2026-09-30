"""Milvus 长期记忆读写（跨会话保存用户关键事实与偏好；批次 14，需求 3.6）。

职责拆分（本文件只保留记忆的读写业务）：
- 集合管理（建集合、字段映射、记录类型）→ app/memory/memory_schema.py
- 写入前的摘要生成（LLM 提炼事实）→ app/memory/memory_summarize.py
- 用户级开关（MySQL）→ app/memory/memory_settings.py

设计红线（用户裁决 + 架构文档 9.3）：
- 读取**必须带 user_id 过滤**：user_id 为空直接抛 ValueError，不存在"查全量"路径；
- 删除为软删除（deleted=true），删除前校验所有权，非本人/不存在一律返回 False
  （API 层统一 404，不暴露存在性）；
- 写入在问答结束后进行，且摘要失败 / 向量失败都不允许影响回答本身（调用方用线程兜底）。
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from app.memory.memory_schema import (
    DEFAULT_COLLECTION_NAME,
    MemoryCollectionMixin,
    MemoryRecord,
)


def _now() -> int:
    """当前 Unix 秒级时间戳：写入时间与过期判定的统一时钟。"""
    return int(time.time())


class LongTermMemoryStore(MemoryCollectionMixin):
    """基于 Milvus 独立集合的长期记忆存储（读写业务；集合管理见 memory_schema）。"""

    def __init__(
        self,
        *,
        milvus_client: Any,
        embedding_client: Any,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        dimension: int = 1024,
        dedup_threshold: float = 0.90,
    ) -> None:
        if not 0.0 < dedup_threshold <= 1.0:
            raise ValueError("dedup_threshold 必须在 (0, 1] 区间")
        self.client = milvus_client
        self.embedding_client = embedding_client
        self.collection_name = collection_name
        self.dimension = dimension
        self.dedup_threshold = dedup_threshold
        self.schema_fields: list[str] = []

    def add_memory(
        self,
        *,
        user_id: str,
        content: str,
        summary: str,
        character_id: str = "",
        importance: float = 0.5,
        source_session_id: str = "",
        expire_at: int = 0,
    ) -> tuple[str, bool]:
        """写入一条记忆；与该用户已有记忆相似度过阈值时更新原记录。

        Returns:
            (memory_id, updated)：updated=True 表示命中去重、更新了原记录。
        Raises:
            ValueError: user_id 为空（写入也必须带归属，防脏数据）。
        """
        if not user_id:
            raise ValueError("user_id 不能为空：长期记忆必须带用户归属")

        now = _now()
        vector = self.embedding_client.embed([summary])[0]

        # 去重：在该用户**未删除且未过期**的记忆里找最相似的
        existing = self._search_rows(user_id, summary, top_k=1)
        if existing:
            hit_distance, hit_row = existing[0]
            if hit_distance >= self.dedup_threshold:
                memory_id = hit_row["memory_id"]
                row = {
                    "memory_id": memory_id,
                    "user_id": hit_row["user_id"],
                    "character_id": character_id or hit_row.get("character_id") or "",
                    "content": content[:4096],
                    "summary": summary[:1024],
                    "importance": max(float(importance), float(hit_row.get("importance") or 0)),
                    "created_at": hit_row["created_at"],  # 创建时间保留
                    "updated_at": now,
                    "expire_at": hit_row.get("expire_at") or 0,
                    "source_session_id": source_session_id
                    or hit_row.get("source_session_id")
                    or "",
                    "deleted": False,  # 相似记忆被重新提起，视作仍有效
                    "dense_vector": vector,
                }
                self.client.upsert(self.collection_name, [row])
                return memory_id, True

        memory_id = uuid.uuid4().hex
        row = {
            "memory_id": memory_id,
            "user_id": user_id,
            "character_id": character_id,
            "content": content[:4096],
            "summary": summary[:1024],
            "importance": float(importance),
            "created_at": now,
            "updated_at": now,
            "expire_at": int(expire_at),
            "source_session_id": source_session_id,
            "deleted": False,
            "dense_vector": vector,
        }
        self.client.upsert(self.collection_name, [row])
        return memory_id, False

    def search(
        self,
        user_id: str,
        query: str,
        *,
        character_id: str | None = None,
        top_k: int = 5,
    ) -> list[MemoryRecord]:
        """按相关度取该用户的记忆，排除已软删与已过期。

        Raises:
            ValueError: user_id 为空——**必须**带用户过滤，不允许查全量。
        """
        if not user_id:
            raise ValueError("user_id 过滤条件缺失：禁止无归属的长期记忆检索")
        rows = self._search_rows(user_id, query, top_k=top_k, character_id=character_id)
        return [self._to_record(row) for _, row in rows]

    def _user_filter(self, user_id: str, character_id: str | None = None) -> str:
        """用户过滤表达式（所有检索路径的必选前缀）。

        过期语义：expire_at == 0 表示永不过期；否则必须大于当前时间。
        """
        expr = f'user_id == "{user_id}" and deleted == false and (expire_at == 0 or expire_at > {_now()})'
        if character_id:
            expr = f'character_id == "{character_id}" and {expr}'
        return expr

    def _search_rows(
        self,
        user_id: str,
        query_text: str,
        *,
        top_k: int,
        character_id: str | None = None,
    ) -> list[tuple[float, dict[str, Any]]]:
        """底层向量检索：返回 [(distance, row)]，按相似度降序。"""
        vector = self.embedding_client.embed([query_text])[0]
        results = self.client.search(
            self.collection_name,
            data=[vector],
            filter=self._user_filter(user_id, character_id),
            limit=top_k,
            output_fields=["*"],
        )
        hits = results[0] if results else []
        return [(float(hit["distance"]), dict(hit["entity"])) for hit in hits]

    def list_memories(
        self,
        *,
        user_id: str,
        character_id: str | None = None,
        page: int = 1,
        page_size: int = 20,
        include_deleted: bool = False,
    ) -> dict[str, Any]:
        """分页列出该用户的记忆（接口 8.1；默认不含已软删）。"""
        if not user_id:
            raise ValueError("user_id 过滤条件缺失：禁止无归属的长期记忆查询")
        expr = self._user_filter(user_id, character_id)
        if include_deleted:
            # 管理口径用：放宽软删过滤，但过期语义保留
            expr = expr.replace(" and deleted == false", "")
        total = len(self.client.query(self.collection_name, filter=expr, output_fields=["memory_id"]))
        offset = (page - 1) * page_size
        rows = self.client.query(
            self.collection_name,
            filter=expr,
            output_fields=["*"],
            limit=page_size,
            offset=offset,
        )
        items = [self._to_record(row).__dict__ for row in rows]
        items.sort(key=lambda r: (-r["updated_at"], r["memory_id"]))
        return {"items": items, "total": total, "page": page, "page_size": page_size}

    def soft_delete(self, *, user_id: str, memory_id: str) -> bool:
        """软删除；先校验所有权。非本人 / 不存在 → False（API 层统一 404）。"""
        if not user_id or not memory_id:
            return False
        rows = self.client.query(
            self.collection_name,
            # 所有权内嵌在过滤表达式里：他人的记忆查不到 → 走"不存在"语义
            filter=f'memory_id == "{memory_id}" and user_id == "{user_id}"',
            output_fields=["*"],
            limit=1,
        )
        if not rows:
            return False
        row = dict(rows[0])
        row["deleted"] = True
        row["updated_at"] = _now()
        self.client.upsert(self.collection_name, [row])
        return True

    def _to_record(self, row: dict[str, Any]) -> MemoryRecord:
        """Milvus 行 → MemoryRecord：None/缺字段统一兜底成类型默认值。"""
        return MemoryRecord(
            memory_id=row["memory_id"],
            user_id=row["user_id"],
            character_id=row.get("character_id") or "",
            content=row.get("content") or "",
            summary=row.get("summary") or "",
            importance=float(row.get("importance") or 0.0),
            created_at=int(row.get("created_at") or 0),
            updated_at=int(row.get("updated_at") or 0),
            expire_at=int(row.get("expire_at") or 0),
            source_session_id=row.get("source_session_id") or "",
            deleted=bool(row.get("deleted")),
        )
