"""批次 22：共用 HTTP 重试工具单测（app/models/http_retry.py）。

覆盖：
- 可重试分类（TimeoutError / 429 / 5xx 可重试，400/401/403 不可重试）
- 超时与 429 后重试成功、4xx 不重试
- 次数耗尽、总预算耗尽
- 指数退避间隔
- request_bytes / request_json 的错误映射（含状态码、超时、JSON 无效、非对象）
"""

import io
import json
import urllib.error

import pytest

from app.models.http_retry import (
    HttpApiError,
    call_with_retry,
    is_retryable,
    request_bytes,
    request_json,
    request_raw,
)


class ScriptedFunc:
    """按脚本逐个抛错 / 返回值；记录调用次数。"""

    def __init__(self, script: list) -> None:
        self.script = list(script)
        self.calls = 0

    def __call__(self):
        self.calls += 1
        action = self.script.pop(0)
        if isinstance(action, Exception):
            raise action
        return action


def run(script: list, **kwargs):
    func = ScriptedFunc(script)
    defaults = dict(
        attempts=3,
        backoff_seconds=0.5,
        total_budget_seconds=15.0,
        what="测试请求",
        sleep=lambda _s: None,
        monotonic=lambda: 0.0,
    )
    defaults.update(kwargs)
    return func, call_with_retry(func, **defaults)


# --------------------------------------------------------------- 分类


def test_is_retryable_timeout_and_429_and_5xx():
    assert is_retryable(TimeoutError("timed out")) is True
    assert is_retryable(HttpApiError("HTTP 429", status_code=429)) is True
    for code in (500, 502, 503, 599):
        assert is_retryable(HttpApiError(f"HTTP {code}", status_code=code)) is True
    for code in (400, 401, 403, 404, 422):
        assert is_retryable(HttpApiError(f"HTTP {code}", status_code=code)) is False
    # 无状态码的 HttpApiError（网络/结构类）不可重试
    assert is_retryable(HttpApiError("network")) is False
    assert is_retryable(ValueError("其他")) is False


# --------------------------------------------------------------- 重试


def test_retry_after_timeout_succeeds():
    func, result = run([TimeoutError("read timed out"), {"ok": 1}])
    assert result == {"ok": 1}
    assert func.calls == 2


def test_retry_after_429_succeeds():
    func, result = run([HttpApiError("HTTP 429", status_code=429), {"ok": 1}])
    assert result == {"ok": 1}
    assert func.calls == 2


def test_http_400_not_retried():
    func = ScriptedFunc([HttpApiError("HTTP 400", status_code=400)])
    with pytest.raises(HttpApiError):
        call_with_retry(
            func,
            attempts=3,
            backoff_seconds=0.5,
            total_budget_seconds=15.0,
            what="测试请求",
            sleep=lambda _s: None,
        )
    assert func.calls == 1  # 不可重试：立即抛，不浪费重试与额度


def test_http_401_not_retried():
    func = ScriptedFunc([HttpApiError("HTTP 401", status_code=401)])
    with pytest.raises(HttpApiError):
        call_with_retry(
            func,
            attempts=3,
            backoff_seconds=0.5,
            total_budget_seconds=15.0,
            what="测试请求",
            sleep=lambda _s: None,
        )
    assert func.calls == 1


def test_exhaust_retries_raises_with_what_in_message():
    func = ScriptedFunc([TimeoutError("timed out")] * 4)
    with pytest.raises(HttpApiError) as excinfo:
        call_with_retry(
            func,
            attempts=3,
            backoff_seconds=0.5,
            total_budget_seconds=15.0,
            what="MinerU 提交任务",
            sleep=lambda _s: None,
        )
    assert func.calls == 4  # 1 次首调 + 3 次重试
    assert "MinerU 提交任务" in str(excinfo.value)


def test_total_budget_stops_retry():
    func = ScriptedFunc([TimeoutError("timed out")] * 5)
    with pytest.raises(HttpApiError):
        call_with_retry(
            func,
            attempts=3,
            backoff_seconds=0.5,
            total_budget_seconds=0.0,
            what="测试请求",
            sleep=lambda _s: None,
        )
    assert func.calls == 1  # 预算为 0：首调失败即放弃


def test_backoff_is_exponential():
    sleeps: list[float] = []
    func, result = run(
        [TimeoutError("t"), TimeoutError("t"), {"ok": 1}],
        backoff_seconds=0.5,
        sleep=sleeps.append,
    )
    assert sleeps == [0.5, 1.0]
    assert func.calls == 3


def test_custom_error_type_used_for_exhaustion():
    class MyError(RuntimeError):
        pass

    func = ScriptedFunc([TimeoutError("t")] * 2)
    with pytest.raises(MyError):
        call_with_retry(
            func,
            attempts=1,
            backoff_seconds=0.5,
            total_budget_seconds=15.0,
            what="测试请求",
            error_type=MyError,
            sleep=lambda _s: None,
        )


# ------------------------------------------------------- 默认传输实现


def _patch_urlopen(monkeypatch, handler):
    monkeypatch.setattr("app.models.http_retry.urllib.request.urlopen", handler)


def test_request_json_parses_body(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout=None):
        captured["method"] = request.get_method()
        captured["url"] = request.full_url
        return io.BytesIO(json.dumps({"code": 0}).encode("utf-8"))

    _patch_urlopen(monkeypatch, fake_urlopen)
    assert request_json("http://x/api", {}, b"{}", 5.0) == {"code": 0}
    assert captured["method"] == "POST"


def test_request_bytes_get_method(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout=None):
        captured["method"] = request.get_method()
        return io.BytesIO(b"raw-bytes")

    _patch_urlopen(monkeypatch, fake_urlopen)
    assert request_bytes("http://x/f.zip", {}, b"", 5.0, method="GET") == b"raw-bytes"
    assert captured["method"] == "GET"


def test_request_json_http_error_carries_status(monkeypatch):
    def fake_urlopen(_request, timeout=None):
        raise urllib.error.HTTPError("u", 403, "Forbidden", {}, None)

    _patch_urlopen(monkeypatch, fake_urlopen)
    with pytest.raises(HttpApiError) as excinfo:
        request_json("http://x/api", {}, b"{}", 5.0)
    assert excinfo.value.status_code == 403


def test_request_json_urlopen_timeout_becomes_timeout_error(monkeypatch):
    def fake_urlopen(_request, timeout=None):
        raise urllib.error.URLError(TimeoutError("timed out"))

    _patch_urlopen(monkeypatch, fake_urlopen)
    with pytest.raises(TimeoutError):
        request_json("http://x/api", {}, b"{}", 5.0)


def test_request_json_network_error_not_timeout(monkeypatch):
    def fake_urlopen(_request, timeout=None):
        raise urllib.error.URLError("connection refused")

    _patch_urlopen(monkeypatch, fake_urlopen)
    with pytest.raises(HttpApiError) as excinfo:
        request_json("http://x/api", {}, b"{}", 5.0)
    assert excinfo.value.status_code is None


def test_request_json_invalid_json(monkeypatch):
    def fake_urlopen(_request, timeout=None):
        return io.BytesIO(b"not json{")

    _patch_urlopen(monkeypatch, fake_urlopen)
    with pytest.raises(HttpApiError):
        request_json("http://x/api", {}, b"{}", 5.0)


def test_request_json_non_object_body(monkeypatch):
    def fake_urlopen(_request, timeout=None):
        return io.BytesIO(b"[1,2,3]")

    _patch_urlopen(monkeypatch, fake_urlopen)
    with pytest.raises(HttpApiError):
        request_json("http://x/api", {}, b"{}", 5.0)


def test_empty_payload_get_does_not_auto_add_content_type(monkeypatch):
    """空 body 的 GET 轮询不得被 urllib 补上 Content-Type / Content-Length: 0。"""
    captured = {}

    def fake_urlopen(request, timeout=None):
        captured["data"] = request.data
        captured["headers"] = dict(request.headers)
        return io.BytesIO(b'{"code":0}')

    _patch_urlopen(monkeypatch, fake_urlopen)
    request_json("http://x/api", {"Authorization": "Bearer k"}, b"", 5.0, method="GET")

    assert captured["data"] is None
    lowered = {key.lower() for key in captured["headers"]}
    assert "content-type" not in lowered
    assert "content-length" not in lowered


# --------------------------------------------------- request_raw（预签名地址）


def _patch_http_client(monkeypatch, captured: dict, status: int = 200, body: bytes = b""):
    class FakeResponse:
        def __init__(self) -> None:
            self.status = status

        def read(self) -> bytes:
            return body

    class FakeConnection:
        def __init__(self, host, port, timeout=None):
            captured["host"] = host
            captured["port"] = port
            captured["timeout"] = timeout

        def request(self, method, target, body=None, headers=None):
            captured["method"] = method
            captured["target"] = target
            captured["body"] = body
            captured["headers"] = dict(headers or {})

        def getresponse(self):
            return FakeResponse()

        def close(self):
            captured["closed"] = True

    monkeypatch.setattr(
        "app.models.http_retry.http.client.HTTPSConnection", FakeConnection
    )


def test_request_raw_sends_only_given_headers(monkeypatch):
    """★ 预签名地址：调用方不给头，就一个头都不发（不能自动补 Content-Type）。"""
    captured: dict = {}
    _patch_http_client(monkeypatch, captured)

    result = request_raw(
        "https://oss.test/bucket/key.pdf?Expires=1&Signature=a%3D",
        method="PUT",
        payload=b"pdf-bytes",
    )

    assert result == b""
    assert captured["method"] == "PUT"
    # query 必须原样透传（签名串在 query 里，重编码就会 SignatureDoesNotMatch）
    assert captured["target"].endswith("?Expires=1&Signature=a%3D")
    assert captured["headers"] == {}
    assert captured["body"] == b"pdf-bytes"
    assert captured["closed"] is True


def test_request_raw_get_without_body(monkeypatch):
    captured: dict = {}
    _patch_http_client(monkeypatch, captured, body=b"zipbytes")

    assert request_raw("https://oss.test/f.zip?sig=1", method="GET") == b"zipbytes"
    assert captured["method"] == "GET"
    assert captured["body"] is None
    assert captured["headers"] == {}


def test_request_raw_forwards_explicit_headers(monkeypatch):
    captured: dict = {}
    _patch_http_client(monkeypatch, captured)

    request_raw("https://oss.test/k", method="PUT", headers={"X-A": "1"})

    assert captured["headers"] == {"X-A": "1"}


def test_request_raw_http_error_carries_status(monkeypatch):
    captured: dict = {}
    _patch_http_client(monkeypatch, captured, status=403)

    with pytest.raises(HttpApiError) as excinfo:
        request_raw("https://oss.test/k", method="PUT")

    assert excinfo.value.status_code == 403


def test_request_raw_500_is_retryable(monkeypatch):
    captured: dict = {}
    _patch_http_client(monkeypatch, captured, status=503)

    with pytest.raises(HttpApiError) as excinfo:
        request_raw("https://oss.test/k", method="PUT")

    assert is_retryable(excinfo.value) is True


def test_request_raw_timeout_is_retryable(monkeypatch):
    captured: dict = {}

    class TimeoutConnection:
        def __init__(self, host, port, timeout=None):
            pass

        def request(self, *args, **kwargs):
            raise TimeoutError("read timed out")

        def close(self):
            pass

    monkeypatch.setattr(
        "app.models.http_retry.http.client.HTTPSConnection", TimeoutConnection
    )
    with pytest.raises(TimeoutError):
        request_raw("https://oss.test/k", method="PUT")


def test_request_raw_rejects_non_http_scheme():
    with pytest.raises(HttpApiError):
        request_raw("file:///etc/passwd", method="GET")


def test_request_raw_connection_error_wrapped(monkeypatch):
    class BrokenConnection:
        def __init__(self, host, port, timeout=None):
            pass

        def request(self, *args, **kwargs):
            raise OSError("connection refused")

        def close(self):
            pass

    monkeypatch.setattr(
        "app.models.http_retry.http.client.HTTPSConnection", BrokenConnection
    )
    with pytest.raises(HttpApiError) as excinfo:
        request_raw("https://oss.test/k", method="PUT")
    assert is_retryable(excinfo.value) is False
