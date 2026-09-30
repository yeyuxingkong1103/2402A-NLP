from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import Boolean, Column, DateTime, Integer, JSON, MetaData, String, Table, Text, delete, insert, select, update
from sqlalchemy.engine import Engine

from backend.app.core.crypto import EncryptedValue
from backend.app.models.memory import ExportJob, MemoryCandidate, MemoryRecord


memory_metadata = MetaData()

memories_table = Table(
    "memories",
    memory_metadata,
    Column("id", String(36), primary_key=True),
    Column("user_id", String(36), nullable=False, index=True),
    Column("encrypted_facts", Text, nullable=False),
    Column("encrypted_summary", Text, nullable=False),
    Column("source_conversation_ids", JSON, nullable=False),
    Column("enabled", Boolean, nullable=False),
    Column("version", Integer, nullable=False),
    Column("prompt_required", Boolean, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    Column("last_used_at", DateTime(timezone=True), nullable=True),
)

memory_candidates_table = Table(
    "memory_candidates",
    memory_metadata,
    Column("id", String(36), primary_key=True),
    Column("memory_id", String(36), nullable=False, index=True),
    Column("user_id", String(36), nullable=False, index=True),
    Column("new_value", JSON, nullable=False),
    Column("status", String(32), nullable=False, index=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("confirmed_at", DateTime(timezone=True), nullable=True),
)

memory_preferences_table = Table(
    "memory_preferences",
    memory_metadata,
    Column("user_id", String(36), primary_key=True),
    Column("auto_extract_enabled", Boolean, nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
)

export_jobs_table = Table(
    "export_jobs",
    memory_metadata,
    Column("id", String(36), primary_key=True),
    Column("user_id", String(36), nullable=False, index=True),
    Column("encrypted_zip", Text, nullable=False),
    Column("password_hmac", String(64), nullable=False),
    Column("expires_at", DateTime(timezone=True), nullable=False, index=True),
    Column("max_downloads", Integer, nullable=False),
    Column("download_count", Integer, nullable=False),
    Column("revoked", Boolean, nullable=False),
    Column("excluded_from_backup", Boolean, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
)


def create_memory_tables(engine: Engine) -> None:
    # 测试和本地验证可显式建表；生产应通过 Alembic 迁移。
    memory_metadata.create_all(engine, checkfirst=True)


def _as_utc(value: datetime | None) -> datetime | None:
    if value is None or value.tzinfo is not None:
        return value
    return value.replace(tzinfo=timezone.utc)


def _serialize_encrypted(value: EncryptedValue) -> str:
    # 加密值以 JSON 信封保存，不拆散密钥版本元数据。
    return json.dumps(value.__dict__, separators=(",", ":"))


def _deserialize_encrypted(value: str) -> EncryptedValue:
    data = json.loads(value)
    return EncryptedValue(
        ciphertext=str(data["ciphertext"]),
        nonce=str(data["nonce"]),
        encrypted_data_key=str(data["encrypted_data_key"]),
        key_version=str(data["key_version"]),
    )


def _memory_values(memory: MemoryRecord) -> dict[str, Any]:
    return {
        "id": memory.id,
        "user_id": memory.user_id,
        "encrypted_facts": _serialize_encrypted(memory.encrypted_facts),
        "encrypted_summary": _serialize_encrypted(memory.encrypted_summary),
        "source_conversation_ids": list(memory.source_conversation_ids),
        "enabled": memory.enabled,
        "version": memory.version,
        "prompt_required": memory.prompt_required,
        "created_at": memory.created_at,
        "updated_at": memory.updated_at,
        "last_used_at": memory.last_used_at,
    }


def _row_to_memory(row) -> MemoryRecord | None:
    if row is None:
        return None
    data = row._mapping
    return MemoryRecord(
        id=data["id"],
        user_id=data["user_id"],
        encrypted_facts=_deserialize_encrypted(data["encrypted_facts"]),
        encrypted_summary=_deserialize_encrypted(data["encrypted_summary"]),
        source_conversation_ids=list(data["source_conversation_ids"]),
        enabled=bool(data["enabled"]),
        version=int(data["version"]),
        prompt_required=bool(data["prompt_required"]),
        created_at=_as_utc(data["created_at"]),
        updated_at=_as_utc(data["updated_at"]),
        last_used_at=_as_utc(data["last_used_at"]),
    )


def _candidate_values(candidate: MemoryCandidate) -> dict[str, Any]:
    return {
        "id": candidate.id,
        "memory_id": candidate.memory_id,
        "user_id": candidate.user_id,
        "new_value": candidate.new_value,
        "status": candidate.status,
        "created_at": candidate.created_at,
        "confirmed_at": candidate.confirmed_at,
    }


def _row_to_candidate(row) -> MemoryCandidate | None:
    if row is None:
        return None
    data = row._mapping
    return MemoryCandidate(
        id=data["id"],
        memory_id=data["memory_id"],
        user_id=data["user_id"],
        new_value=dict(data["new_value"]),
        status=data["status"],
        created_at=_as_utc(data["created_at"]),
        confirmed_at=_as_utc(data["confirmed_at"]),
    )


def _export_values(job: ExportJob) -> dict[str, Any]:
    return {
        "id": job.id,
        "user_id": job.user_id,
        "encrypted_zip": _serialize_encrypted(job.encrypted_zip),
        "password_hmac": job.password_hmac,
        "expires_at": job.expires_at,
        "max_downloads": job.max_downloads,
        "download_count": job.download_count,
        "revoked": job.revoked,
        "excluded_from_backup": job.excluded_from_backup,
        "created_at": job.created_at,
    }


def _row_to_export(row) -> ExportJob | None:
    if row is None:
        return None
    data = row._mapping
    return ExportJob(
        id=data["id"],
        user_id=data["user_id"],
        encrypted_zip=_deserialize_encrypted(data["encrypted_zip"]),
        password_hmac=data["password_hmac"],
        expires_at=_as_utc(data["expires_at"]),
        max_downloads=int(data["max_downloads"]),
        download_count=int(data["download_count"]),
        revoked=bool(data["revoked"]),
        excluded_from_backup=bool(data["excluded_from_backup"]),
        created_at=_as_utc(data["created_at"]),
    )


class SQLAlchemyMemoryRepository:
    # 长期记忆、候选更新和导出作业共用同一个 SQL 引擎。
    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    def save_memory(self, memory: MemoryRecord) -> None:
        values = _memory_values(memory)
        with self.engine.begin() as connection:
            exists = connection.execute(select(memories_table.c.id).where(memories_table.c.id == memory.id)).first()
            if exists:
                connection.execute(update(memories_table).where(memories_table.c.id == memory.id).values(**values))
                return
            connection.execute(insert(memories_table).values(**values))

    def get_memory(self, memory_id: str, user_id: str) -> MemoryRecord | None:
        with self.engine.begin() as connection:
            row = connection.execute(select(memories_table).where(memories_table.c.id == memory_id).where(memories_table.c.user_id == user_id)).first()
        return _row_to_memory(row)

    def get_memory_by_id(self, memory_id: str) -> MemoryRecord | None:
        with self.engine.begin() as connection:
            row = connection.execute(select(memories_table).where(memories_table.c.id == memory_id)).first()
        return _row_to_memory(row)

    def list_memories(self, user_id: str) -> list[MemoryRecord]:
        with self.engine.begin() as connection:
            rows = connection.execute(select(memories_table).where(memories_table.c.user_id == user_id).order_by(memories_table.c.created_at)).all()
        return [memory for row in rows if (memory := _row_to_memory(row)) is not None]

    def delete_memory(self, memory_id: str, user_id: str) -> bool:
        with self.engine.begin() as connection:
            result = connection.execute(delete(memories_table).where(memories_table.c.id == memory_id).where(memories_table.c.user_id == user_id))
        return result.rowcount > 0

    def set_auto_extract_enabled(self, user_id: str, enabled: bool) -> None:
        values = {"user_id": user_id, "auto_extract_enabled": enabled, "updated_at": datetime.now(timezone.utc)}
        with self.engine.begin() as connection:
            exists = connection.execute(select(memory_preferences_table.c.user_id).where(memory_preferences_table.c.user_id == user_id)).first()
            if exists:
                connection.execute(update(memory_preferences_table).where(memory_preferences_table.c.user_id == user_id).values(**values))
                return
            connection.execute(insert(memory_preferences_table).values(**values))

    def is_auto_extract_enabled(self, user_id: str) -> bool:
        with self.engine.begin() as connection:
            value = connection.execute(select(memory_preferences_table.c.auto_extract_enabled).where(memory_preferences_table.c.user_id == user_id)).scalar_one_or_none()
        return True if value is None else bool(value)

    def save_candidate(self, candidate: MemoryCandidate) -> None:
        values = _candidate_values(candidate)
        with self.engine.begin() as connection:
            exists = connection.execute(select(memory_candidates_table.c.id).where(memory_candidates_table.c.id == candidate.id)).first()
            if exists:
                connection.execute(update(memory_candidates_table).where(memory_candidates_table.c.id == candidate.id).values(**values))
                return
            connection.execute(insert(memory_candidates_table).values(**values))

    def get_candidate(self, candidate_id: str) -> MemoryCandidate | None:
        with self.engine.begin() as connection:
            row = connection.execute(select(memory_candidates_table).where(memory_candidates_table.c.id == candidate_id)).first()
        return _row_to_candidate(row)

    def save_export_job(self, job: ExportJob) -> None:
        values = _export_values(job)
        with self.engine.begin() as connection:
            exists = connection.execute(select(export_jobs_table.c.id).where(export_jobs_table.c.id == job.id)).first()
            if exists:
                connection.execute(update(export_jobs_table).where(export_jobs_table.c.id == job.id).values(**values))
                return
            connection.execute(insert(export_jobs_table).values(**values))

    def get_export_job(self, job_id: str, user_id: str) -> ExportJob | None:
        with self.engine.begin() as connection:
            row = connection.execute(select(export_jobs_table).where(export_jobs_table.c.id == job_id).where(export_jobs_table.c.user_id == user_id)).first()
        return _row_to_export(row)

    def get_export_job_by_id(self, job_id: str) -> ExportJob | None:
        with self.engine.begin() as connection:
            row = connection.execute(select(export_jobs_table).where(export_jobs_table.c.id == job_id)).first()
        return _row_to_export(row)

    def delete_memory_data(self, user_id: str) -> dict[str, int]:
        with self.engine.begin() as connection:
            candidates = connection.execute(delete(memory_candidates_table).where(memory_candidates_table.c.user_id == user_id)).rowcount
            memories = connection.execute(delete(memories_table).where(memories_table.c.user_id == user_id)).rowcount
            prefs = connection.execute(delete(memory_preferences_table).where(memory_preferences_table.c.user_id == user_id)).rowcount
        return {"candidates": candidates, "memories": memories, "preferences": prefs}

    def delete_export_data(self, user_id: str) -> dict[str, int]:
        with self.engine.begin() as connection:
            exports = connection.execute(delete(export_jobs_table).where(export_jobs_table.c.user_id == user_id)).rowcount
        return {"exports": exports}

    def cleanup_export_jobs(self, expired_before: datetime) -> dict[str, int]:
        # 导出密文属于短期敏感数据，过期或撤销后应由运维任务及时清理。
        cutoff = _as_utc(expired_before)
        with self.engine.begin() as connection:
            exports = connection.execute(
                delete(export_jobs_table).where((export_jobs_table.c.expires_at <= cutoff) | (export_jobs_table.c.revoked.is_(True)))
            ).rowcount
        return {"exports": exports}

    def delete_user_data(self, user_id: str) -> dict[str, int]:
        with self.engine.begin() as connection:
            candidates = connection.execute(delete(memory_candidates_table).where(memory_candidates_table.c.user_id == user_id)).rowcount
            memories = connection.execute(delete(memories_table).where(memories_table.c.user_id == user_id)).rowcount
            exports = connection.execute(delete(export_jobs_table).where(export_jobs_table.c.user_id == user_id)).rowcount
            prefs = connection.execute(delete(memory_preferences_table).where(memory_preferences_table.c.user_id == user_id)).rowcount
        return {"candidates": candidates, "memories": memories, "exports": exports, "preferences": prefs}
