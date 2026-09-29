import time
from collections import deque

from backend.app.core.config import settings
from backend.app.schemas.chat import ChatResult


class InMemoryAccountRateLimiter:
    # MVP 服务内账号频率限制；生产可替换为 Redis 等集中式实现。
    def __init__(self) -> None:
        self.minute_windows: dict[str, deque[float]] = {}
        self.day_windows: dict[str, deque[float]] = {}

    def reset(self) -> None:
        self.minute_windows.clear()
        self.day_windows.clear()

    def check(self, user_id: str, conversation_id: str) -> ChatResult | None:
        now = time.time()
        minute_window = self.minute_windows.setdefault(user_id, deque())
        day_window = self.day_windows.setdefault(user_id, deque())
        self._drop_expired(minute_window, now, 60)
        self._drop_expired(day_window, now, 86400)
        if len(minute_window) >= settings.ACCOUNT_MESSAGES_PER_MINUTE:
            return ChatResult("rate_limited", conversation_id, "", "当前账号每分钟消息数已达上限，请稍后再试。", reason="account_minute_rate_limit")
        if len(day_window) >= settings.ACCOUNT_MESSAGES_PER_DAY:
            return ChatResult("rate_limited", conversation_id, "", "当前账号今日消息数已达上限，请明日再试或联系官方渠道。", reason="account_day_rate_limit")
        minute_window.append(now)
        day_window.append(now)
        return None

    def _drop_expired(self, window: deque[float], now: float, ttl_seconds: int) -> None:
        while window and now - window[0] >= ttl_seconds:
            window.popleft()
