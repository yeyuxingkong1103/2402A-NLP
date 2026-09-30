from datetime import timedelta

from sqlalchemy import create_engine, select
from sqlalchemy.pool import StaticPool

from backend.app.core.crypto import encrypt_text
from backend.app.core.security import utc_now
from backend.app.models.memory import ExportJob, MemoryCandidate, MemoryRecord
from backend.app.repositories.memory_repository import SQLAlchemyMemoryRepository, create_memory_tables, memories_table


def _repository(monkeypatch):
    monkeypatch.setenv("APP_MASTER_KEY", "test-master-key-32-bytes-minimum")
    monkeypatch.setenv("APP_HMAC_KEY", "test-hmac-key-32-bytes-minimum")
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool, future=True)
    create_memory_tables(engine)
    return SQLAlchemyMemoryRepository(engine), engine


def _memory(user_id="user-1"):
    now = utc_now()
    return MemoryRecord(
        id="mem-1",
        user_id=user_id,
        encrypted_facts=encrypt_text('{"phone":"[手机号]"}', purpose="memory-facts"),
        encrypted_summary=encrypt_text("脱敏摘要", purpose="memory-summary"),
        source_conversation_ids=["c1"],
        enabled=True,
        version=1,
        created_at=now,
        updated_at=now,
        prompt_required=True,
    )


def test_memory_repository_round_trips_encrypted_memory(monkeypatch):
    repository, engine = _repository(monkeypatch)
    memory = _memory()

    repository.save_memory(memory)
    restored = repository.get_memory("mem-1", "user-1")

    assert restored == memory
    with engine.begin() as connection:
        stored = connection.execute(select(memories_table.c.encrypted_summary)).scalar_one()
    assert "脱敏摘要" not in stored


def test_memory_repository_enforces_owner_and_preferences(monkeypatch):
    repository, _ = _repository(monkeypatch)
    repository.save_memory(_memory(user_id="owner"))

    assert repository.get_memory("mem-1", "other") is None
    assert repository.list_memories("other") == []
    assert repository.is_auto_extract_enabled("owner") is True
    repository.set_auto_extract_enabled("owner", False)
    assert repository.is_auto_extract_enabled("owner") is False


def test_memory_repository_round_trips_candidate_and_export_job(monkeypatch):
    repository, _ = _repository(monkeypatch)
    repository.save_memory(_memory())
    candidate = MemoryCandidate(id="cand-1", memory_id="mem-1", user_id="user-1", new_value={"facts": {"child": "yes"}}, status="pending", created_at=utc_now())
    job = ExportJob(
        id="exp-1",
        user_id="user-1",
        encrypted_zip=encrypt_text("zip-bytes", purpose="user-export"),
        password_hmac="password-hmac",
        expires_at=utc_now() + timedelta(hours=24),
        created_at=utc_now(),
    )

    repository.save_candidate(candidate)
    repository.save_export_job(job)

    assert repository.get_candidate("cand-1") == candidate
    assert repository.get_export_job("exp-1", "user-1") == job
    assert repository.get_export_job("exp-1", "other") is None


def test_memory_repository_cleans_expired_and_revoked_exports(monkeypatch):
    repository, _ = _repository(monkeypatch)
    now = utc_now()
    active = ExportJob(id="exp-active", user_id="user-1", encrypted_zip=encrypt_text("active", purpose="user-export"), password_hmac="hash", expires_at=now + timedelta(hours=1), created_at=now)
    expired = ExportJob(id="exp-expired", user_id="user-1", encrypted_zip=encrypt_text("expired", purpose="user-export"), password_hmac="hash", expires_at=now - timedelta(seconds=1), created_at=now)
    revoked = ExportJob(id="exp-revoked", user_id="user-1", encrypted_zip=encrypt_text("revoked", purpose="user-export"), password_hmac="hash", expires_at=now + timedelta(hours=1), revoked=True, created_at=now)
    for job in (active, expired, revoked):
        repository.save_export_job(job)

    deleted = repository.cleanup_export_jobs(now)

    assert deleted == {"exports": 2}
    assert repository.get_export_job("exp-active", "user-1") == active
    assert repository.get_export_job("exp-expired", "user-1") is None
    assert repository.get_export_job("exp-revoked", "user-1") is None


    repository, _ = _repository(monkeypatch)
    repository.save_memory(_memory())
    repository.save_candidate(MemoryCandidate(id="cand-1", memory_id="mem-1", user_id="user-1", new_value={}, status="pending", created_at=utc_now()))
    repository.set_auto_extract_enabled("user-1", False)
    repository.save_export_job(ExportJob(id="exp-1", user_id="user-1", encrypted_zip=encrypt_text("zip", purpose="user-export"), password_hmac="hash", expires_at=utc_now() + timedelta(hours=1), created_at=utc_now()))

    deleted = repository.delete_user_data("user-1")

    assert deleted == {"candidates": 1, "memories": 1, "exports": 1, "preferences": 1}
    assert repository.list_memories("user-1") == []
