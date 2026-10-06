from backend.app.core.crypto import decrypt_text, encrypt_text, hmac_digest


def test_encrypt_decrypt_round_trip(monkeypatch):
    monkeypatch.setenv("APP_MASTER_KEY", "test-master-key-32-bytes-minimum-value")
    encrypted = encrypt_text("用户手机号13800138000", purpose="message")

    assert encrypted.ciphertext != "用户手机号13800138000"
    assert decrypt_text(encrypted, purpose="message") == "用户手机号13800138000"


def test_hmac_digest_is_stable_and_not_plain_hash(monkeypatch):
    monkeypatch.setenv("APP_HMAC_KEY", "test-hmac-key")
    first = hmac_digest("13800138000", purpose="phone")
    second = hmac_digest("13800138000", purpose="phone")

    assert first == second
    assert "13800138000" not in first
