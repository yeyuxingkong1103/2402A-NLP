import io
import json
import logging
import secrets
import uuid
import zipfile
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Protocol

from backend.app.api.v1.chat import chat_service
from backend.app.core.crypto import encrypt_text, hmac_digest
from backend.app.core.security import utc_now
from backend.app.models.memory import ExportJob
from backend.app.services.memory_service import list_user_memories, memory_to_export_dict
from backend.app.services.privacy_service import redact_pii

logger = logging.getLogger(__name__)
_export_jobs: dict[str, ExportJob] = {}
_second_factor_tokens: dict[str, "SecondFactorToken"] = {}
_one_time_passwords: dict[str, str] = {}
_password_claimed: set[str] = set()
_repository: "ExportRepository | None" = None


class ExportRepository(Protocol):
    # 导出服务只持久化作业元数据；短期 token 和一次性口令仍保留在进程内。
    def save_export_job(self, job: ExportJob) -> None: ...
    def get_export_job(self, job_id: str, user_id: str) -> ExportJob | None: ...
    def get_export_job_by_id(self, job_id: str) -> ExportJob | None: ...
    def delete_export_data(self, user_id: str) -> dict[str, int]: ...
    def cleanup_export_jobs(self, expired_before: Any) -> dict[str, int]: ...


@dataclass
class SecondFactorToken:
    # 服务端保存二次验证状态，客户端只能提交短期 token，不能自称已验证。
    user_id: str
    expires_at: Any
    used: bool = False


def set_export_repository(repository: ExportRepository | None) -> None:
    # 生产启动时显式注入 SQL repository；默认内存路径继续服务测试和 MVP。
    global _repository
    _repository = repository


def reset_export_jobs_for_tests() -> None:
    # 测试专用清理导出作业和二次验证状态。
    _export_jobs.clear()
    _second_factor_tokens.clear()
    _one_time_passwords.clear()
    _password_claimed.clear()
    set_export_repository(None)


def cleanup_expired_export_jobs(expired_before: Any | None = None) -> dict[str, int]:
    # 运维清理入口删除过期或已撤销导出作业，并同步移除进程内一次性口令。
    cutoff = expired_before or utc_now()
    if _repository is not None:
        deleted = _repository.cleanup_export_jobs(cutoff)
        _one_time_passwords.clear()
        _password_claimed.clear()
        logger.info("expired export jobs cleaned", extra={"deleted_jobs": deleted.get("exports", 0), "repository": True})
        return deleted
    removable = {
        job_id
        for job_id, job in _export_jobs.items()
        if job.revoked or job.expires_at <= cutoff
    }
    for job_id in removable:
        _export_jobs.pop(job_id, None)
        _one_time_passwords.pop(job_id, None)
        _password_claimed.discard(job_id)
    logger.info("expired export jobs cleaned", extra={"deleted_jobs": len(removable), "repository": False})
    return {"exports": len(removable)}


def delete_export_data_for_user(user_id: str) -> dict[str, int]:
    # 注销时删除导出作业、二次验证 token 和临时口令，避免继续访问个人数据。
    if _repository is not None:
        deleted_jobs = _repository.delete_export_data(user_id).get("exports", 0)
        job_ids = set()
    else:
        job_ids = {job_id for job_id, job in _export_jobs.items() if job.user_id == user_id}
        deleted_jobs = sum(1 for job_id in job_ids if _export_jobs.pop(job_id, None) is not None)
    deleted_passwords = sum(1 for job_id in job_ids if _one_time_passwords.pop(job_id, None) is not None)
    for job_id in job_ids:
        _password_claimed.discard(job_id)
    deleted_tokens = sum(
        1
        for token, challenge in list(_second_factor_tokens.items())
        if challenge.user_id == user_id and _second_factor_tokens.pop(token, None) is not None
    )
    logger.info(
        "user export data deleted",
        extra={"user_id": user_id, "deleted_jobs": deleted_jobs, "deleted_passwords": deleted_passwords, "deleted_tokens": deleted_tokens},
    )
    return {"jobs": deleted_jobs, "passwords": deleted_passwords, "tokens": deleted_tokens}


def issue_second_factor_token(user_id: str) -> str:
    # MVP 中复用已登录身份后签发短期导出挑战 token；生产可替换为短信/OTP 校验。
    token = secrets.token_urlsafe(32)
    _second_factor_tokens[token] = SecondFactorToken(user_id=user_id, expires_at=utc_now() + timedelta(minutes=5))
    logger.info("export second factor token issued", extra={"user_id": user_id})
    return token


def _consume_second_factor_token(user_id: str, token: str | None) -> None:
    challenge = _second_factor_tokens.get(token or "")
    if challenge is None or challenge.user_id != user_id or challenge.used or challenge.expires_at <= utc_now():
        raise PermissionError("导出用户数据需要服务端二次验证")
    challenge.used = True


def build_export_payload(user_id: str) -> dict[str, Any]:
    # 导出只包含用户可携带数据，明确排除审计日志、内部 Prompt、密钥等内部数据。
    conversations = []
    messages = []
    for conversation in chat_service.conversations.values():
        if conversation.user_id != user_id:
            continue
        conversations.append(
            {
                "id": conversation.id,
                "active": conversation.active,
                "follow_up_count": conversation.follow_up_count,
                "message_ids": list(conversation.message_ids),
            }
        )
        for message_id in conversation.message_ids:
            message = chat_service.messages.get(message_id)
            if message is None or message.user_id != user_id:
                continue
            messages.append(
                {
                    "id": message.id,
                    "conversation_id": message.conversation_id,
                    "user_text": redact_pii(message.user_text).text,
                    "answer": redact_pii(message.answer).text,
                    "status": message.status,
                }
            )
    memories = [memory_to_export_dict(memory) for memory in list_user_memories(user_id)]
    return {"user_id": user_id, "conversations": conversations, "messages": messages, "memories": memories}


def _build_markdown(payload: dict[str, Any]) -> str:
    # Markdown 方便用户阅读，内容仍来源于脱敏后的导出 payload。
    lines = ["# 用户数据导出", "", f"用户ID：{payload['user_id']}", "", "## 会话"]
    for conversation in payload["conversations"]:
        lines.append(f"- {conversation['id']}：{len(conversation['message_ids'])} 条消息")
    lines.extend(["", "## 长期记忆"])
    for memory in payload["memories"]:
        lines.append(f"- {memory['id']}：{memory['summary']}")
    return "\n".join(lines) + "\n"


def _zip_payload(payload: dict[str, Any], password: str) -> bytes:
    # Python 标准库只支持读取加密 ZIP；这里把随机口令写入注释并由外层 AES 加密保护。
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.comment = f"password-required:{password}".encode("utf-8")
        archive.writestr("data.json", json.dumps(payload, ensure_ascii=False, indent=2))
        archive.writestr("data.md", _build_markdown(payload))
    return buffer.getvalue()


def export_user_data(user_id: str, second_factor_token: str | None = None) -> ExportJob:
    # 导出前必须消费服务端二次验证 token，避免登录态被盗后批量拉取数据。
    _consume_second_factor_token(user_id, second_factor_token)
    payload = build_export_payload(user_id)
    password = secrets.token_urlsafe(18)
    zip_bytes = _zip_payload(payload, password)
    encrypted_zip = encrypt_text(zip_bytes.decode("latin1"), purpose="user-export")
    now = utc_now()
    job = ExportJob(id=f"exp-{uuid.uuid4().hex}", user_id=user_id, encrypted_zip=encrypted_zip, password_hmac=hmac_digest(password, purpose="export-password"), expires_at=now + timedelta(hours=24), created_at=now)
    if _repository is not None:
        _repository.save_export_job(job)
    else:
        _export_jobs[job.id] = job
    _one_time_passwords[job.id] = password
    _password_claimed.discard(job.id)
    logger.info("user export created", extra={"user_id": user_id, "job_id": job.id, "expires_at": job.expires_at.isoformat(), "max_downloads": job.max_downloads})
    return job


def get_export_job(job_id: str) -> ExportJob:
    # 测试和下载入口按 ID 获取作业，不返回不存在作业的伪数据。
    if _repository is not None:
        job = _repository.get_export_job_by_id(job_id)
        if job is None:
            raise KeyError(job_id)
        return job
    return _export_jobs[job_id]


def _save_export_job(job: ExportJob) -> None:
    if _repository is not None:
        _repository.save_export_job(job)
    else:
        _export_jobs[job.id] = job


def claim_export_password(job_id: str, user_id: str, second_factor_token: str | None = None) -> str:
    # 口令独立领取且只返回一次，避免普通下载 JSON 同时携带密文和解密材料。
    _consume_second_factor_token(user_id, second_factor_token)
    job = _repository.get_export_job(job_id, user_id) if _repository is not None else _export_jobs.get(job_id)
    if job is None or job.user_id != user_id:
        raise PermissionError("导出文件不存在或无权领取口令")
    if job.revoked or job.expires_at <= utc_now():
        raise PermissionError("导出口令已过期或已撤销")
    if job.id in _password_claimed:
        raise PermissionError("导出口令已领取")
    password = _one_time_passwords.get(job.id)
    if not password:
        raise PermissionError("导出口令不可用")
    _password_claimed.add(job.id)
    logger.info("user export password claimed", extra={"user_id": user_id, "job_id": job.id})
    return password


def download_export_job(job_id: str, user_id: str) -> dict[str, Any]:
    # 下载时强制归属、24 小时有效期、撤销状态和最多 3 次下载限制；不返回 ZIP 口令。
    job = _repository.get_export_job(job_id, user_id) if _repository is not None else _export_jobs.get(job_id)
    if job is None or job.user_id != user_id:
        raise PermissionError("导出文件不存在或无权下载")
    if job.revoked or job.expires_at <= utc_now():
        raise PermissionError("导出文件已过期或已撤销")
    if job.download_count >= job.max_downloads:
        raise PermissionError("导出文件下载次数已达上限")
    job.download_count += 1
    _save_export_job(job)
    logger.info("user export downloaded", extra={"user_id": user_id, "job_id": job.id, "download_count": job.download_count})
    return {"encrypted_zip": job.encrypted_zip.__dict__, "password_delivery": "claim_required"}


def revoke_export_job(job_id: str, user_id: str) -> None:
    # 用户撤销后立即禁止继续下载。
    job = _repository.get_export_job(job_id, user_id) if _repository is not None else _export_jobs.get(job_id)
    if job is None or job.user_id != user_id:
        raise PermissionError("导出文件不存在或无权撤销")
    job.revoked = True
    _save_export_job(job)
    _one_time_passwords.pop(job.id, None)
    _password_claimed.discard(job.id)
