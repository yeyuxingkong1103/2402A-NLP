from backend.app.core.config import AppSettings
from backend.app.services.request_control import InMemoryRequestController, RedisRequestController


class FakeRedis:
    def __init__(self) -> None:
        self.values: dict[str, int] = {}
        self.ttls: dict[str, int] = {}

    def incr(self, key: str) -> int:
        self.values[key] = self.values.get(key, 0) + 1
        return self.values[key]

    def decr(self, key: str) -> int:
        self.values[key] = self.values.get(key, 0) - 1
        return self.values[key]

    def expire(self, key: str, ttl_seconds: int) -> None:
        self.ttls[key] = ttl_seconds


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _settings(**overrides) -> AppSettings:
    base = AppSettings()
    for key, value in overrides.items():
        setattr(base, key, value)
    return base


def test_in_memory_request_controller_enforces_rate_and_releases_concurrency():
    clock = FakeClock()
    controller = InMemoryRequestController(_settings(ACCOUNT_MESSAGES_PER_MINUTE=1), now_provider=clock)

    result, lease = controller.try_enter("user-1")
    assert result is None
    assert lease is not None
    limited, denied_lease = controller.try_enter("user-1")

    assert limited.reason == "account_minute_rate_limit"
    assert denied_lease is None
    lease.release()
    clock.now += 61
    result, lease = controller.try_enter("user-1")
    assert result is None
    assert lease is not None


def test_redis_request_controller_shares_limits_and_releases_active_keys():
    redis = FakeRedis()
    controller = RedisRequestController(redis, _settings(ACCOUNT_CONCURRENT_REQUESTS=1, SYSTEM_CONCURRENT_REQUESTS=2, ACCOUNT_MESSAGES_PER_MINUTE=10), key_prefix="test")

    result, lease = controller.try_enter("user-1")
    assert result is None
    assert lease is not None
    limited, denied_lease = controller.try_enter("user-1")

    assert limited.reason == "account_concurrency_limit"
    assert denied_lease is None
    lease.release()
    assert redis.values["test:active:account:user-1"] == 0
    assert redis.values["test:active:system"] == 0
    assert redis.ttls["test:rate:minute:user-1"] == 60
