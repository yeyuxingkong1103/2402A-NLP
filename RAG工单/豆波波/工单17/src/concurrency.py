from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager


class RequestLimiter:
    def __init__(self, limit: int = 16):
        if limit < 1:
            raise ValueError("limit must be positive")
        self._semaphore = asyncio.Semaphore(limit)

    @asynccontextmanager
    async def slot(self):
        await self._semaphore.acquire()
        try:
            yield
        finally:
            self._semaphore.release()


API_REQUEST_LIMITER = RequestLimiter(16)
