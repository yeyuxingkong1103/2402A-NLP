from datetime import timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from backend.app.repositories.memory_repository import SQLAlchemyMemoryRepository, create_memory_tables
from backend.app.services import export_service, memory_service
from backend.app.services.export_service import claim_export_password, cleanup_expired_export_jobs, download_export_job, export_user_data, issue_second_factor_token, revoke_export_job
from backend.app.services.memory_service import create_memory_from_conversation, list_user_memories, mark_candidate_update, confirm_memory_update
from backend.app.core.crypto import decrypt_text
from backend.app.core.security import utc_now


@pytest.fixture()
def memory_repository_env(monkeypatch):
    monkeypatch.setenv("APP_MASTER_KEY", "test-master-key-32-bytes-minimum-value")
    monkeypatch.setenv("APP_HMAC_KEY", "test-hmac-key-32-bytes-minimum-value")
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool, future=True)
    create_memory_tables(engine)
    repository = SQLAlchemyMemoryRepository(engine)
    memory_service.set_memory_repository(repository)
    export_service.set_export_repository(repository)
    export_service._second_factor_tokens.clear()
    export_service._one_time_passwords.clear()
    yield repository
    memory_service.reset_memory_store_for_tests()
    export_service.reset_export_jobs_for_tests()


def test_memory_service_uses_repository_for_memory_and_candidates(memory_repository_env):
    memory = create_memory_from_conversation("user-1", "c1", {"phone": "13800138000"}, "手机号13800138000摘要")

    restored = memory_repository_env.get_memory(memory.id, "user-1")
    assert restored is not None
    assert decrypt_text(restored.encrypted_summary, purpose="memory-summary") == "手机号[手机号]摘要"
    assert [item.id for item in list_user_memories("user-1")] == [memory.id]

    candidate = mark_candidate_update(memory.id, {"summary": "新摘要", "facts": {"case_type": "divorce"}})
    updated = confirm_memory_update(candidate.id, "user-1")

    assert updated.version == 2
    assert memory_repository_env.get_candidate(candidate.id).status == "confirmed"


def test_export_service_persists_job_updates_and_revocation(memory_repository_env):
    create_memory_from_conversation("user-1", "c1", {"case_type": "rent"}, "租赁摘要")
    token = issue_second_factor_token("user-1")

    job = export_user_data("user-1", second_factor_token=token)
    first = download_export_job(job.id, "user-1")
    password_token = issue_second_factor_token("user-1")
    password = claim_export_password(job.id, "user-1", password_token)
    second = download_export_job(job.id, "user-1")
    restored = memory_repository_env.get_export_job(job.id, "user-1")

    assert first["password_delivery"] == "claim_required"
    assert second["password_delivery"] == "claim_required"
    assert password
    assert restored.download_count == 2

    revoke_export_job(job.id, "user-1")
    assert memory_repository_env.get_export_job(job.id, "user-1").revoked is True
    with pytest.raises(PermissionError):
        download_export_job(job.id, "user-1")


def test_export_cleanup_removes_expired_repository_jobs(memory_repository_env):
    token = issue_second_factor_token("user-1")
    job = export_user_data("user-1", second_factor_token=token)
    job.expires_at = utc_now() - timedelta(seconds=1)
    memory_repository_env.save_export_job(job)

    deleted = cleanup_expired_export_jobs()

    assert deleted == {"exports": 1}
    assert memory_repository_env.get_export_job(job.id, "user-1") is None


    memory = create_memory_from_conversation("user-1", "c1", {"case_type": "rent"}, "租赁摘要")
    token = issue_second_factor_token("user-1")
    job = export_user_data("user-1", second_factor_token=token)

    deleted_memories = memory_service.delete_all_memories_for_user("user-1")

    assert deleted_memories == 1
    assert memory_repository_env.get_memory(memory.id, "user-1") is None
    assert memory_repository_env.get_export_job(job.id, "user-1") is not None

    deleted_exports = export_service.delete_export_data_for_user("user-1")

    assert deleted_exports["jobs"] == 1
    assert memory_repository_env.get_export_job(job.id, "user-1") is None
