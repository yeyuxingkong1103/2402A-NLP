"""单元测试：src.core.security 密码哈希与 JWT。"""
import pytest
import jwt as pyjwt

from src.core import security
from src.core.config import settings


# ---------------- bcrypt ----------------
def test_hash_password_is_not_plaintext_and_verifies():
    password = "Test123456"
    hashed = security.hash_password(password)
    assert hashed != password
    assert hashed.startswith("$2")
    assert security.verify_password(password, hashed) is True


def test_verify_password_wrong_password():
    hashed = security.hash_password("Test123456")
    assert security.verify_password("WrongPassword", hashed) is False
    assert security.verify_password("", hashed) is False
    assert security.verify_password("Test123456", "") is False


def test_hash_password_uses_salt():
    assert security.hash_password("SamePassword") != security.hash_password("SamePassword")


def test_verify_password_invalid_hash_returns_false():
    assert security.verify_password("Test123456", "not-a-bcrypt-hash") is False


# ---------------- JWT ----------------
def test_access_token_roundtrip():
    token = security.create_access_token(123, roles=["user"])
    payload = security.decode_token(token)
    assert payload["sub"] == "123"
    assert payload["type"] == "access"
    assert payload["roles"] == ["user"]
    assert payload["exp"] > payload["iat"]


def test_refresh_token_type():
    payload = security.decode_token(security.create_refresh_token(7))
    assert payload["sub"] == "7"
    assert payload["type"] == "refresh"
    # refresh token 不携带角色
    assert "roles" not in payload


def test_token_expires_in():
    assert security.token_expires_in() == settings.jwt_access_token_expire_minutes * 60


def test_expired_token_raises():
    expired = security._create_token(1, "access", -1)
    with pytest.raises(pyjwt.ExpiredSignatureError):
        security.decode_token(expired)


def test_token_signed_with_other_secret_raises():
    forged = pyjwt.encode({"sub": "1", "type": "access"}, "another-secret",
                          algorithm=settings.jwt_algorithm)
    with pytest.raises(pyjwt.PyJWTError):
        security.decode_token(forged)


def test_tampered_token_raises():
    token = security.create_access_token(1)
    with pytest.raises(pyjwt.PyJWTError):
        security.decode_token(token[:-3] + "abc")


def test_malformed_token_raises():
    with pytest.raises(pyjwt.PyJWTError):
        security.decode_token("this-is-not-a-jwt")


def test_access_and_refresh_tokens_are_distinguishable():
    access = security.decode_token(security.create_access_token(9))
    refresh = security.decode_token(security.create_refresh_token(9))
    assert access["type"] != refresh["type"]
    assert access["sub"] == refresh["sub"] == "9"