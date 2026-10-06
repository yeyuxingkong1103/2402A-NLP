from datetime import datetime, timedelta, timezone
import hashlib
import hmac
import secrets

from fastapi import HTTPException


PUBLIC_COLLECTIONS = [
    "civil_code_articles",
    "civil_interpretations",
    "civil_cases",
    "civil_elements",
    "civil_evidence",
    "civil_processes",
    "civil_questions",
    "civil_citations",
]

EXTENDED_PUBLIC_COLLECTIONS = [
    "civil_evidence",
    "civil_processes",
    "civil_questions",
    "civil_citations",
]

PUBLIC_COLLECTION_LABELS = {
    "civil_code_articles": "民法典条文库",
    "civil_interpretations": "民法典司法解释库",
    "civil_cases": "民法典案例库",
    "civil_elements": "民法典法律要件库",
    "civil_evidence": "民法典证据规则库",
    "civil_processes": "民法典处理流程库",
    "civil_questions": "民法典问题模板库",
    "civil_citations": "民法典引用关系库",
}

PRIVATE_COLLECTION_PREFIX = "user_upload_documents"


def private_collection_name(embedding_dim: int | None = None) -> str:
    if embedding_dim is None:
        return PRIVATE_COLLECTION_PREFIX
    return f"{PRIVATE_COLLECTION_PREFIX}_{int(embedding_dim)}"

PBKDF2_ITERATIONS = 260_000


def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        PBKDF2_ITERATIONS,
    ).hex()
    return f"pbkdf2_sha256${PBKDF2_ITERATIONS}${salt}${digest}"


def verify_password(password: str, stored: str) -> bool:
    parts = stored.split("$")
    if len(parts) == 4 and parts[0] == "pbkdf2_sha256":
        _, iterations, salt, digest = parts
        try:
            count = int(iterations)
        except ValueError:
            return False
        expected = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt.encode("utf-8"),
            count,
        ).hex()
        return hmac.compare_digest(expected, digest)
    if len(parts) == 2:
        salt, digest = parts
        expected = hashlib.sha256((salt + password).encode("utf-8")).hexdigest()
        return hmac.compare_digest(expected, digest)
    return False


def new_token() -> str:
    return secrets.token_urlsafe(32)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def expires_after(seconds: int) -> str:
    return (utc_now() + timedelta(seconds=seconds)).isoformat()


def require_owner(user_id: str, owner_id: str) -> None:
    if user_id != owner_id:
        raise HTTPException(status_code=403, detail="没有权限访问该资源")
