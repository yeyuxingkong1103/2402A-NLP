"""Pure ASGI request-body size enforcement."""

from __future__ import annotations

import os
from typing import Any, Awaitable, Callable


class RequestBodyTooLarge(Exception):
    """Internal signal used before an HTTP response has started."""


class RequestSizeLimitMiddleware:
    """Limit actual streamed bytes, including bodies without Content-Length."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(
        self,
        scope: dict[str, Any],
        receive: Callable[[], Awaitable[dict[str, Any]]],
        send: Callable[[dict[str, Any]], Awaitable[None]],
    ) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        limit = int(os.getenv("MAX_REQUEST_SIZE_MB", "10")) * 1024 * 1024
        headers = {key.lower(): value for key, value in scope.get("headers", [])}
        declared = headers.get(b"content-length")
        try:
            if declared is not None and int(declared) > limit:
                await self._reject(send)
                return
        except ValueError:
            await self._reject(send)
            return

        total = 0
        response_started = False

        async def bounded_receive() -> dict[str, Any]:
            nonlocal total
            message = await receive()
            if message.get("type") == "http.request":
                total += len(message.get("body", b""))
                if total > limit:
                    raise RequestBodyTooLarge
            return message

        async def tracked_send(message: dict[str, Any]) -> None:
            nonlocal response_started
            if message.get("type") == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, bounded_receive, tracked_send)
        except RequestBodyTooLarge:
            if not response_started:
                await self._reject(send)

    @staticmethod
    async def _reject(send: Callable[[dict[str, Any]], Awaitable[None]]) -> None:
        body = b'{"error":"request_too_large","detail":"Request body is too large"}'
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode("ascii")),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})
