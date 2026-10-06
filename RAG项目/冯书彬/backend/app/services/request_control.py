import time
from dataclasses import dataclass
from typing import Protocol

from backend.app.core.config import AppSettings, settings


@dataclass
class RequestControlResult:
    # 控制器只返回原因和用户提示，调用方决定如何转换成具体响应对象。
    reason: str
    message: str


class RequestLease(Protocol):
    # lease 负责在请求结束时释放并发占用，Redis 和内存实现保持同一语义。
    def release(self) -> None: ...


class RequestController(Protocol):
    def reset(self) -> None: ...
    def try_enter(self, user_id: str) -> tuple[RequestControlResult | None, RequestLease | None]: ...


class NoopLease:
    def release(self) -> None:
        return None


class InMemoryRequestLease:
    def __init__(self, controller: "InMemoryRequestController", user_id: str) -> None:
        self.controller = controller
        self.user_id = user_id
        self.released = False

    def release(self) -> None:
        if self.released:
            return
        self.released = True
        self.controller.leave(self.user_id)


class InMemoryRequestController:
    # 默认单进程控制器，保留 MVP 测试的确定性行为。
    def __init__(self, app_settings: AppSettings = settings, now_provider=time.time) -> None:
        self.settings = app_settings
        self.now_provider = now_provider
        self.minute_windows: dict[str, list[float]] = {}
        self.day_windows: dict[str, list[float]] = {}
        self.account_active: dict[str, int] = {}
        self.system_active = 0

    def reset(self) -> None:
        self.minute_windows.clear()
        self.day_windows.clear()
        self.account_active.clear()
        self.system_active = 0

    def try_enter(self, user_id: str) -> tuple[RequestControlResult | None, RequestLease | None]:
        if self.account_active.get(user_id, 0) >= self.settings.ACCOUNT_CONCURRENT_REQUESTS:
            return _account_concurrency_limit(), None
        if self.system_active >= self.settings.SYSTEM_CONCURRENT_REQUESTS:
            return _system_concurrency_limit(), None
        rate_result = self._check_rate(user_id)
        if rate_result is not None:
            return rate_result, None
        self.account_active[user_id] = self.account_active.get(user_id, 0) + 1
        self.system_active += 1
        return None, InMemoryRequestLease(self, user_id)

    def leave(self, user_id: str) -> None:
        self.account_active[user_id] = max(0, self.account_active.get(user_id, 1) - 1)
        self.system_active = max(0, self.system_active - 1)

    def _check_rate(self, user_id: str) -> RequestControlResult | None:
        now = self.now_provider()
        minute_window = self.minute_windows.setdefault(user_id, [])
        day_window = self.day_windows.setdefault(user_id, [])
        _drop_expired(minute_window, now, 60)
        _drop_expired(day_window, now, 86400)
        if len(minute_window) >= self.settings.ACCOUNT_MESSAGES_PER_MINUTE:
            return RequestControlResult("account_minute_rate_limit", "当前账号每分钟消息数已达上限，请稍后再试。")
        if len(day_window) >= self.settings.ACCOUNT_MESSAGES_PER_DAY:
            return RequestControlResult("account_day_rate_limit", "当前账号今日消息数已达上限，请明日再试或联系官方渠道。")
        minute_window.append(now)
        day_window.append(now)
        return None


class RedisRequestLease:
    def __init__(self, client, keys: list[str]) -> None:
        self.client = client
        self.keys = keys
        self.released = False

    def release(self) -> None:
        if self.released:
            return
        self.released = True
        for key in self.keys:
            try:
                self.client.decr(key)
            except Exception:
                continue


class RedisRequestController:
    # Redis 后端使用原子 INCR/EXPIRE，让限流和并发在多进程间保持一致。
    def __init__(self, client, app_settings: AppSettings = settings, key_prefix: str = "myrag:request") -> None:
        self.client = client
        self.settings = app_settings
        self.key_prefix = key_prefix.rstrip(":")

    def reset(self) -> None:
        return None

    def try_enter(self, user_id: str) -> tuple[RequestControlResult | None, RequestLease | None]:
        rate_result = self._check_rate(user_id)
        if rate_result is not None:
            return rate_result, None
        account_key = f"{self.key_prefix}:active:account:{user_id}"
        system_key = f"{self.key_prefix}:active:system"
        account_count = self._increment_with_ttl(account_key, self.settings.ANSWER_TIMEOUT_SECONDS + 30)
        if account_count > self.settings.ACCOUNT_CONCURRENT_REQUESTS:
            self.client.decr(account_key)
            return _account_concurrency_limit(), None
        system_count = self._increment_with_ttl(system_key, self.settings.ANSWER_TIMEOUT_SECONDS + 30)
        if system_count > self.settings.SYSTEM_CONCURRENT_REQUESTS:
            self.client.decr(account_key)
            self.client.decr(system_key)
            return _system_concurrency_limit(), None
        return None, RedisRequestLease(self.client, [account_key, system_key])

    def _check_rate(self, user_id: str) -> RequestControlResult | None:
        minute_key = f"{self.key_prefix}:rate:minute:{user_id}"
        day_key = f"{self.key_prefix}:rate:day:{user_id}"
        minute_count = self._increment_with_ttl(minute_key, 60)
        if minute_count > self.settings.ACCOUNT_MESSAGES_PER_MINUTE:
            return RequestControlResult("account_minute_rate_limit", "当前账号每分钟消息数已达上限，请稍后再试。")
        day_count = self._increment_with_ttl(day_key, 86400)
        if day_count > self.settings.ACCOUNT_MESSAGES_PER_DAY:
            return RequestControlResult("account_day_rate_limit", "当前账号今日消息数已达上限，请明日再试或联系官方渠道。")
        return None

    def _increment_with_ttl(self, key: str, ttl_seconds: int) -> int:
        value = int(self.client.incr(key))
        if value == 1:
            self.client.expire(key, ttl_seconds)
        return value


def create_request_controller(app_settings: AppSettings = settings) -> RequestController:
    backend = app_settings.REQUEST_CONTROL_BACKEND.strip().lower()
    if backend == "memory":
        return InMemoryRequestController(app_settings)
    if backend != "redis":
        raise ValueError("unsupported REQUEST_CONTROL_BACKEND")
    if not app_settings.REDIS_URL:
        raise ValueError("REDIS_URL is required for redis request control")
    import redis

    return RedisRequestController(redis.from_url(app_settings.REDIS_URL, decode_responses=True), app_settings)


def _account_concurrency_limit() -> RequestControlResult:
    return RequestControlResult("account_concurrency_limit", "当前账号并发请求过多，请稍后再试。")


def _system_concurrency_limit() -> RequestControlResult:
    return RequestControlResult("system_concurrency_limit", "系统繁忙，请稍后再试。")


def _drop_expired(window: list[float], now: float, ttl_seconds: int) -> None:
    while window and now - window[0] >= ttl_seconds:
        window.pop(0)
