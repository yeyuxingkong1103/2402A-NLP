"""认证测试：口令散列、令牌签名与过期、角色授权。"""

from __future__ import annotations

import time

import pytest

from role_rag.api.auth import AuthContext, TokenCodec, hash_password, verify_password
from role_rag.errors import AuthError, ValidationError


def test_password_hash_roundtrip():
    stored = hash_password("alice123", iterations=1000)
    assert stored.startswith("pbkdf2_sha256$1000$")
    assert verify_password("alice123", stored)
    assert not verify_password("alice124", stored)


def test_password_hash_is_salted():
    assert hash_password("same", 1000) != hash_password("same", 1000)


def test_empty_password_rejected():
    with pytest.raises(ValidationError):
        hash_password("", 1000)


def test_verify_handles_broken_stored_value():
    assert verify_password("x", "garbage") is False
    assert verify_password("x", "md5$1$a$b") is False


def test_token_roundtrip():
    codec = TokenCodec("secret-key", ttl=60)
    token = codec.encode({"username": "alice", "roles": ["lawyer"]})
    payload = codec.decode(token)
    assert payload["username"] == "alice"
    assert payload["roles"] == ["lawyer"]
    assert payload["exp"] - payload["iat"] == 60


def test_token_tamper_detected():
    codec = TokenCodec("secret-key", ttl=60)
    token = codec.encode({"username": "alice"})
    body, signature = token.split(".")
    with pytest.raises(AuthError):
        codec.decode(f"{body}x.{signature}")
    with pytest.raises(AuthError):
        codec.decode(f"{body}.{signature[:-1]}x")


def test_token_signed_with_other_secret_rejected():
    with pytest.raises(AuthError):
        TokenCodec("another-secret", ttl=60).decode(TokenCodec("secret-key", ttl=60).encode({"username": "a"}))


def test_token_expiry():
    codec = TokenCodec("secret-key", ttl=1)
    token = codec.encode({"username": "alice"}, ttl=-1)
    with pytest.raises(AuthError):
        codec.decode(token)


def test_token_format_validation():
    codec = TokenCodec("secret-key")
    with pytest.raises(AuthError):
        codec.decode("not-a-token")


def test_auth_context_role_permissions():
    user = AuthContext(username="bob", roles=["lawyer", "scientist"])
    assert user.can_use_role("lawyer")
    assert not user.can_use_role("financial_planner")
    admin = AuthContext(username="admin", roles=["*"], is_admin=True)
    assert admin.can_use_role("financial_planner")


def test_visible_roles_filtered():
    user = AuthContext(username="carol", roles=["financial_planner"])
    enabled = ["financial_planner", "scientist", "lawyer"]
    assert user.visible_roles(enabled) == ["financial_planner"]
    assert AuthContext(username="admin", roles=["*"], is_admin=True).visible_roles(enabled) == enabled
