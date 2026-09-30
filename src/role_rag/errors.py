"""统一异常体系：所有对外错误都带稳定 code 与 HTTP 状态码。"""

from __future__ import annotations

from typing import Any


class RoleRagError(Exception):
    """业务异常基类。"""

    code = "role_rag_error"
    http_status = 500

    def __init__(self, message: str, **detail: Any) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.detail:
            payload["detail"] = self.detail
        return payload


class ConfigError(RoleRagError):
    code = "config_error"
    http_status = 500


class ValidationError(RoleRagError):
    code = "validation_error"
    http_status = 400


class AuthError(RoleRagError):
    code = "auth_error"
    http_status = 401


class AccessDeniedError(RoleRagError):
    code = "access_denied"
    http_status = 403


class NotFoundError(RoleRagError):
    code = "not_found"
    http_status = 404


class RateLimitError(RoleRagError):
    code = "rate_limited"
    http_status = 429


class IngestError(RoleRagError):
    code = "ingest_error"
    http_status = 500


class DependencyError(RoleRagError):
    """Milvus / Redis / 模型等外部依赖不可用。"""

    code = "dependency_unavailable"
    http_status = 503
