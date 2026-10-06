from datetime import datetime, timedelta, timezone

from backend.app.repositories.otp_repository import RedisOtpStore
from backend.app.services.auth_store import InMemoryAuthStore, OtpState


class FakeRedis:
    def __init__(self):
        self.values = {}
        self.ttls = {}

    def get(self, key):
        return self.values.get(key)

    def setex(self, key, seconds, value):
        self.values[key] = value
        self.ttls[key] = seconds

    def delete(self, key):
        self.values.pop(key, None)
        self.ttls.pop(key, None)


def test_redis_otp_store_round_trips_state_and_sets_remaining_ttl():
    now = datetime(2026, 9, 17, 10, 0, tzinfo=timezone.utc)
    client = FakeRedis()
    store = RedisOtpStore(client, now_provider=lambda: now)
    state = OtpState(
        code_hash="hashed-code",
        expires_at=now + timedelta(seconds=300),
        attempts=2,
        locked_until=now + timedelta(seconds=60),
    )

    store.save("otp-key", state)
    restored = store.get("otp-key")

    assert client.ttls["otp-key"] == 300
    assert restored == state
    assert "hashed-code" in client.values["otp-key"]


def test_redis_otp_store_removes_expired_state_without_writing():
    now = datetime(2026, 9, 17, 10, 0, tzinfo=timezone.utc)
    client = FakeRedis()
    store = RedisOtpStore(client, now_provider=lambda: now)

    store.save("expired-key", OtpState("hash", now - timedelta(seconds=1)))

    assert client.values == {}
    assert store.get("expired-key") is None


def test_redis_otp_store_ignores_corrupt_payload():
    client = FakeRedis()
    client.values["broken-key"] = "not-json"

    assert RedisOtpStore(client).get("broken-key") is None


def test_auth_store_delegates_otp_operations_to_redis_backend():
    now = datetime(2026, 9, 17, 10, 0, tzinfo=timezone.utc)
    client = FakeRedis()
    backend = RedisOtpStore(client, now_provider=lambda: now)
    store = InMemoryAuthStore()
    store.set_otp_backend(backend)
    state = OtpState("hash", now + timedelta(seconds=300))

    store.save_otp("otp-key", state)

    assert store.get_otp("otp-key") == state
    store.delete_otp("otp-key")
    assert store.get_otp("otp-key") is None
