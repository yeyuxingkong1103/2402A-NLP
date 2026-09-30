"""API-key authentication and server-derived request identities."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import threading
import time
from collections import defaultdict, deque
from contextvars import ContextVar
from dataclasses import dataclass

from fastapi import Request
from fastapi.responses import JSONResponse


@dataclass(frozen=True)
class AuthSettings:
    """Runtime security settings loaded without exposing secret values."""

    required: bool


@dataclass(frozen=True)
class AuthContext:
    """Caller identity derived from server-side credentials, never request data."""

    user_id: int
    tenant_id: int
    authenticated: bool


_current_auth: ContextVar[AuthContext | None] = ContextVar("current_auth", default=None)


def settings() -> AuthSettings:
    """Read settings for every request so test and runtime changes take effect."""
    return AuthSettings(
        required=os.getenv("REQUIRE_API_KEY", "false").lower() == "true"
        or os.getenv("APP_ENV", "development") == "production",
    )


def _provided_key(request: Request) -> str:
    """Accept the documented header and the standard bearer form."""
    value = request.headers.get("X-API-Key", "").strip()
    if value:
        return value
    authorization = request.headers.get("Authorization", "")
    prefix = "Bearer "
    return authorization[len(prefix) :].strip() if authorization.startswith(prefix) else ""


def _default_context(*, authenticated: bool = False) -> AuthContext:
    return AuthContext(
        user_id=int(os.getenv("DEFAULT_USER_ID", "1001")),
        tenant_id=int(os.getenv("DEFAULT_TENANT_ID", "1")),
        authenticated=authenticated,
    )


def _configured_identities() -> list[tuple[str, AuthContext]]:
    """Load single-key and optional multi-tenant credential mappings."""
    identities: list[tuple[str, AuthContext]] = []
    default_key = os.getenv("API_KEY") or ""
    if default_key:
        identities.append((default_key, _default_context(authenticated=True)))

    raw = os.getenv("API_KEY_IDENTITIES", "").strip()
    if not raw:
        return identities
    try:
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError("mapping required")
        for key, identity in payload.items():
            if not isinstance(key, str) or not key or not isinstance(identity, dict):
                raise ValueError("invalid identity entry")
            identities.append(
                (
                    key,
                    AuthContext(
                        user_id=int(identity["user_id"]),
                        tenant_id=int(identity["tenant_id"]),
                        authenticated=True,
                    ),
                )
            )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError("API_KEY_IDENTITIES must be a valid identity mapping") from exc
    return identities


def _resolve_identity(provided: str) -> AuthContext | None:
    for expected, identity in _configured_identities():
        if hmac.compare_digest(provided, expected):
            return identity
    return None


def active_auth_context() -> AuthContext | None:
    """Return the active HTTP identity, if execution is inside middleware."""
    return _current_auth.get()


def current_auth_context() -> AuthContext:
    """Return the request identity, or the local-development default."""
    return active_auth_context() or _default_context()


def validate_production_security() -> None:
    """Fail closed when production starts without an authentication secret."""
    if os.getenv("APP_ENV", "development") == "production" and not _configured_identities():
        raise RuntimeError("Production requires API_KEY or API_KEY_IDENTITIES")


_rate_lock = threading.Lock()
_rate_windows: dict[str, deque[float]] = defaultdict(deque)


def _rate_key(request: Request) -> str:
    """Avoid retaining raw API keys in the process rate-limit map."""
    provided = _provided_key(request)
    identity = f"key:{provided}" if provided else f"ip:{request.client.host if request.client else 'unknown'}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _allowed_by_rate_limit(request: Request, limit: int) -> bool:
    """Apply a bounded fixed-window rate limit in one process."""
    if limit <= 0:
        return True
    now = time.monotonic()
    cutoff = now - 60.0
    key = _rate_key(request)
    with _rate_lock:
        window = _rate_windows[key]
        while window and window[0] <= cutoff:
            window.popleft()
        if len(window) >= limit:
            return False
        window.append(now)
        if len(_rate_windows) > 10_000:
            stale = [item for item, values in _rate_windows.items() if not values or values[-1] <= cutoff]
            for item in stale:
                _rate_windows.pop(item, None)
        return True


async def security_middleware(request: Request, call_next):
    """Authenticate the request and install a server-derived identity."""
    config = settings()
    path = request.url.path

    is_public = path in {"/health", "/"} or path.startswith("/static/")
    provided = _provided_key(request)
    configured = _configured_identities()
    identity = _resolve_identity(provided) if provided else None
    if config.required and not is_public and identity is None:
        return JSONResponse(
            status_code=401,
            content={"error": "unauthorized", "detail": "Valid X-API-Key is required"},
        )
    if not is_public and provided and configured and identity is None:
        return JSONResponse(
            status_code=401,
            content={"error": "unauthorized", "detail": "Invalid API credential"},
        )
    if not is_public and not _allowed_by_rate_limit(
        request, int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
    ):
        return JSONResponse(
            status_code=429,
            headers={"Retry-After": "60"},
            content={"error": "rate_limited", "detail": "Too many requests"},
        )

    token = _current_auth.set(identity or _default_context())
    try:
        return await call_next(request)
    finally:
        _current_auth.reset(token)
