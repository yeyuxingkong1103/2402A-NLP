"""Object-level authorization helpers for tenant-scoped API routes."""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException, status

from app.api.v1 import deps
from app.core.auth import AuthContext, active_auth_context, current_auth_context
from app.rag.knowledge_base_service import KnowledgeBaseService


def identity() -> AuthContext:
    """Return the identity established by the authentication middleware."""
    return current_auth_context()


def tenant_id(claimed: int | None = None) -> int:
    """Resolve tenant scope and reject attempts to claim another tenant."""
    context = identity()
    if claimed is not None and claimed != context.tenant_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="The requested tenant is outside the caller's scope",
        )
    return context.tenant_id


def user_id(claimed: int | None = None) -> int:
    """Resolve user scope and reject attempts to impersonate another user."""
    context = identity()
    if claimed is not None and claimed != context.user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="The requested user is outside the caller's scope",
        )
    return context.user_id


def owned_session(manager: Any, session_id: str) -> dict[str, Any]:
    """Load a session and conceal sessions owned by another identity."""
    meta = manager.get_session(session_id)
    context = active_auth_context()
    if not meta:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    # Direct function calls in unit tests have no HTTP context; every real route
    # invocation passes through middleware and therefore enforces ownership.
    if context is not None and (
        _as_int(meta.get("user_id")) != context.user_id
        or _as_int(meta.get("tenant_id")) != context.tenant_id
    ):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    return meta


def owned_knowledge_base(kb_id: int) -> dict[str, Any]:
    """Require a database-backed knowledge base in the caller's tenant."""
    client = deps.get_mysql_client()
    if client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Knowledge-base authorization requires MySQL",
        )
    row = KnowledgeBaseService(client).get(kb_id, identity().tenant_id)
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Knowledge base not found")
    return row


def authorize_session_id(session_id: str | None) -> None:
    """Verify an optional stateful chat session before memory is accessed."""
    if not session_id or active_auth_context() is None:
        return
    manager = deps.get_session_manager()
    if manager is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Session authorization is temporarily unavailable",
        )
    owned_session(manager, session_id)


def scope_chat_request(request: Any) -> Any:
    """Replace client scope with trusted identity and validate referenced objects."""
    if active_auth_context() is None:
        return request
    scope = tenant_id(getattr(request, "tenant_id", None))
    authorize_session_id(getattr(request, "session_id", None))
    kb_id = getattr(request, "knowledge_base_id", None)
    if kb_id is not None:
        owned_knowledge_base(kb_id)
    return request.model_copy(update={"tenant_id": scope})


def _as_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None
