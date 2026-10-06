"""MongoDB 存储：原文归档、长期记忆摘要、对话归档（可选）。

未启用或连接失败时，所有方法安全返回空值，不影响主流程。
"""
from __future__ import annotations

from typing import Optional

from app.config import settings
from app.logging_conf import log


class MongoStore:
    def __init__(self) -> None:
        self._client = None
        self._db = None

    def _conn(self):
        if self._db is not None:
            return self._db
        if not settings.mongo_enabled:
            raise RuntimeError("MongoDB 未启用")
        from pymongo import MongoClient

        self._client = MongoClient(settings.mongo_uri, serverSelectionTimeoutMS=3000)
        self._client.admin.command("ping")
        self._db = self._client[settings.mongo_db]
        log.info("MongoDB 连接成功: %s", settings.mongo_db)
        return self._db

    @property
    def enabled(self) -> bool:
        try:
            self._conn()
            return True
        except Exception as exc:  # noqa: BLE001
            log.warning("MongoDB 不可用: %s", exc)
            return False

    # ===== 原文归档 =====
    def save_documents(self, docs: list[dict]) -> int:
        if not docs:
            return 0
        try:
            return len(self._conn()["documents"].insert_many(docs).inserted_ids)
        except Exception as exc:  # noqa: BLE001
            log.warning("Mongo insert 失败: %s", exc)
            return 0

    def get_documents(self, doc_id: str, limit: int = 100) -> list[dict]:
        try:
            cur = self._conn()["documents"].find({"doc_id": doc_id}, {"_id": 0}).limit(limit)
            return list(cur)
        except Exception:  # noqa: BLE001
            return []

    def delete_documents(self, doc_id: str) -> int:
        try:
            return self._conn()["documents"].delete_many({"doc_id": doc_id}).deleted_count
        except Exception:  # noqa: BLE001
            return 0

    # ===== 长期记忆摘要 =====
    def save_summary(self, user_id: str, role_id: int, summary: str) -> None:
        try:
            self._conn()["memory"].update_one(
                {"user_id": user_id, "role_id": role_id},
                {"$set": {"summary": summary}},
                upsert=True,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("Mongo save_summary 失败: %s", exc)

    def get_summary(self, user_id: str, role_id: int) -> Optional[str]:
        try:
            row = self._conn()["memory"].find_one({"user_id": user_id, "role_id": role_id})
            return row.get("summary") if row else None
        except Exception:  # noqa: BLE001
            return None

    # ===== 对话归档 =====
    def archive_messages(self, user_id: str, role_id: int, messages: list[dict]) -> None:
        if not messages:
            return
        try:
            self._conn()["chat_archive"].insert_one(
                {"user_id": user_id, "role_id": role_id, "messages": messages}
            )
        except Exception:  # noqa: BLE001
            pass

    def search_memory(self, query: str, limit: int = 5) -> list[dict]:
        try:
            cur = (
                self._conn()["memory"]
                .find({"$text": {"$search": query}}, {"_id": 0})
                .limit(limit)
            )
            return list(cur)
        except Exception:  # noqa: BLE001
            return []


mongo_store = MongoStore()
