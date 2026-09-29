"""批次 22：MinerU 客户端单测（app/models/mineru.py）。

全部用替身传输，不联网。覆盖：
- 正常路径：batch_id → 上传 → 轮询 done → 取 full.md
- ★ 上传请求绝不携带 Authorization（带了会被对象存储 403）
- 轮询状态流转 pending/running/done、轮询超时、failed 态
- 结果包缺 full.md、正文为空
- 401 不重试、429 重试
- 页数取值：优先 extract_progress.total_pages，缺失时退化为页图数量
- 文件不存在 / 超过体积上限 → complete=False 且不抛异常
"""

import io
import json
import zipfile
from pathlib import Path

import pytest

from app.models.http_retry import HttpApiError
from app.models.mineru import MineruClient

PNG_BYTES = b"\x89PNG\r\n\x1a\nfake"


class ScriptedJsonTransport:
    """按脚本返回 JSON 响应；记录每次调用的 url/headers/payload。"""

    def __init__(self, script: list) -> None:
        self.script = list(script)
        self.calls: list[tuple[str, dict, bytes]] = []

    def __call__(self, url, headers, payload, timeout):
        self.calls.append((url, dict(headers), payload))
        action = self.script.pop(0)
        if isinstance(action, Exception):
            raise action
        return action


class RecordingUpload:
    """记录上传请求（url/headers/字节数）。"""

    def __init__(self, error: Exception | None = None) -> None:
        self.calls: list[tuple[str, dict, int]] = []
        self.error = error

    def __call__(self, url, headers, payload, timeout):
        self.calls.append((url, dict(headers), len(payload)))
        if self.error is not None:
            raise self.error
        return b""


class StaticDownload:
    """返回固定 zip 字节（或抛错）。"""

    def __init__(self, data: bytes = b"", error: Exception | None = None) -> None:
        self.data = data
        self.error = error
        self.calls = 0

    def __call__(self, url, headers, payload, timeout):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.data


class FakeTime:
    """可控时钟：sleep 推进时间，便于验证轮询超时与退避。"""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def make_zip(files: dict[str, bytes | str]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return buffer.getvalue()


def batch_response(batch_id: str = "b-1") -> dict:
    return {
        "code": 0,
        "data": {"batch_id": batch_id, "file_urls": ["https://upload.test/presigned"]},
    }


def result_response(state: str, *, pages: int | None = 10, err_msg: str = "") -> dict:
    entry: dict = {
        "file_name": "x.pdf",
        "state": state,
        "err_msg": err_msg,
        "full_zip_url": "https://cdn.test/full.zip" if state == "done" else "",
    }
    if pages is not None:
        entry["extract_progress"] = {"extracted_pages": pages, "total_pages": pages}
    return {"code": 0, "data": {"batch_id": "b-1", "extract_result": [entry]}}


@pytest.fixture()
def pdf_file(tmp_path: Path) -> Path:
    path = tmp_path / "sample.pdf"
    path.write_bytes(b"%PDF-1.4 fake pdf bytes")
    return path


def make_client(script: list, zip_bytes: bytes = b"", **kwargs):
    transport = ScriptedJsonTransport(script)
    upload = RecordingUpload()
    download = StaticDownload(zip_bytes)
    clock = FakeTime()
    options = dict(
        api_base_url="https://mineru.test",
        api_key="test-key",
        poll_interval_seconds=3.0,
        max_poll_seconds=300.0,
        transport_json=transport,
        upload=upload,
        download=download,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )
    options.update(kwargs)  # 允许用例覆盖轮询参数
    client = MineruClient(**options)
    return client, transport, upload, download, clock


# ------------------------------------------------------------- 正常路径


def test_parse_returns_markdown_from_full_md(pdf_file):
    archive = make_zip({"full.md": "# 标题\n\n第一条 正文内容。", "page_1.png": PNG_BYTES})
    client, transport, upload, download, clock = make_client(
        [batch_response(), result_response("done", pages=10)], archive
    )

    result = client.parse(pdf_file)

    assert result["complete"] is True
    assert result["error"] == ""
    assert "第一条 正文内容。" in result["content"]
    assert result["page_count"] == 10
    # 调用序列：POST 申请 → (上传) → GET 轮询
    assert transport.calls[0][0].endswith("/api/v4/file-urls/batch")
    assert "extract-results/batch/b-1" in transport.calls[1][0]
    assert upload.calls[0][0] == "https://upload.test/presigned"
    assert download.calls == 1
    # 轨迹可供验收取证
    assert client.last_trace["batch_id"] == "b-1"
    assert client.last_trace["final_state"] == "done"
    assert client.last_trace["complete"] is True


def test_upload_request_sends_no_headers_at_all(pdf_file):
    """★ 预签名地址按「空 Content-Type」签名：

    多带一个 Content-Type（或 Authorization）都会被 OSS 判为
    SignatureDoesNotMatch 而 403。所以上传请求头必须是空的。
    """
    archive = make_zip({"full.md": "正文"})
    client, _transport, upload, _download, _clock = make_client(
        [batch_response(), result_response("done")], archive
    )

    client.parse(pdf_file)

    _url, headers, size = upload.calls[0]
    assert headers == {}
    assert size == len(b"%PDF-1.4 fake pdf bytes")


def test_default_upload_transport_does_not_add_content_type(pdf_file, monkeypatch):
    """默认上传实现（request_raw + http.client）不得补 Content-Type。"""
    captured: dict = {}

    class FakeResponse:
        status = 200

        def read(self) -> bytes:
            return b""

    class FakeConnection:
        def __init__(self, host, port, timeout=None):
            captured["host"] = host

        def request(self, method, target, body=None, headers=None):
            captured["method"] = method
            captured["headers"] = dict(headers or {})

        def getresponse(self):
            return FakeResponse()

        def close(self):
            captured["closed"] = True

    monkeypatch.setattr(
        "app.models.http_retry.http.client.HTTPSConnection", FakeConnection
    )
    client, _t, _u, _d, _c = make_client([batch_response()])
    client.upload = MineruClient._request_upload.__get__(client)

    client._upload_file("https://mineru.oss.test/presigned?sig=x", pdf_file, {"events": []})

    assert captured["method"] == "PUT"
    assert captured["headers"] == {}  # 关键：一个头都不发
    assert "Content-Type" not in captured["headers"]
    assert captured["closed"] is True


def test_poll_states_flow_pending_running_done(pdf_file):
    archive = make_zip({"full.md": "正文"})
    client, transport, _upload, _download, clock = make_client(
        [batch_response(), result_response("pending"), result_response("running"), result_response("done")],
        archive,
    )

    result = client.parse(pdf_file)

    assert result["complete"] is True
    states = [e["state"] for e in client.last_trace["events"] if e["step"] == "poll"]
    assert states == ["pending", "running", "done"]
    assert client.last_trace["polls"] == 3
    # 前两次轮询后各睡一个轮询间隔
    assert clock.sleeps == [3.0, 3.0]


def test_page_count_falls_back_to_image_count(pdf_file):
    archive = make_zip(
        {"full.md": "正文", "a.png": PNG_BYTES, "b.png": PNG_BYTES, "c.jpg": PNG_BYTES}
    )
    client, _t, _u, _d, _c = make_client(
        [batch_response(), result_response("done", pages=None)], archive
    )

    result = client.parse(pdf_file)

    assert result["complete"] is True
    assert result["page_count"] == 3


def test_page_count_from_layout_json(pdf_file):
    """MinerU v4 实测不返回 extract_progress，页数取 layout.json 的 pdf_info 长度。"""
    layout = json.dumps({"pdf_info": [{"page_idx": i} for i in range(27)]})
    archive = make_zip({"full.md": "正文", "layout.json": layout})
    client, _t, _u, _d, _c = make_client(
        [batch_response(), result_response("done", pages=None)], archive
    )

    result = client.parse(pdf_file)

    assert result["page_count"] == 27
    sources = [e for e in client.last_trace["events"] if e["step"] == "page_count"]
    assert sources[-1]["source"] == "layout.json"


def test_page_count_from_nested_layout_json(pdf_file):
    layout = json.dumps({"pdf_info": [{"page_idx": i} for i in range(5)]})
    archive = make_zip({"out/full.md": "正文", "out/layout.json": layout})
    client, _t, _u, _d, _c = make_client(
        [batch_response(), result_response("done", pages=None)], archive
    )

    assert client.parse(pdf_file)["page_count"] == 5


def test_page_count_from_origin_pdf(pdf_file):
    """layout.json 缺失时用包内 *_origin.pdf 数页（需 PyMuPDF）。"""
    fitz = pytest.importorskip("fitz")
    document = fitz.open()
    for _ in range(4):
        document.new_page()
    origin_pdf = document.tobytes()
    document.close()
    archive = make_zip({"full.md": "正文", "abc_origin.pdf": origin_pdf})
    client, _t, _u, _d, _c = make_client(
        [batch_response(), result_response("done", pages=None)], archive
    )

    result = client.parse(pdf_file)

    assert result["page_count"] == 4
    sources = [e for e in client.last_trace["events"] if e["step"] == "page_count"]
    assert sources[-1]["source"] == "origin.pdf"


def test_page_count_prefers_extract_progress(pdf_file):
    layout = json.dumps({"pdf_info": [{"page_idx": i} for i in range(99)]})
    archive = make_zip({"full.md": "正文", "layout.json": layout})
    client, _t, _u, _d, _c = make_client(
        [batch_response(), result_response("done", pages=27)], archive
    )

    result = client.parse(pdf_file)

    assert result["page_count"] == 27  # 接口给的值优先于包内推断
    sources = [e for e in client.last_trace["events"] if e["step"] == "page_count"]
    assert sources[-1]["source"] == "extract_progress"


def test_broken_layout_json_falls_through(pdf_file):
    archive = make_zip(
        {"full.md": "正文", "layout.json": "not json{", "a.png": PNG_BYTES}
    )
    client, _t, _u, _d, _c = make_client(
        [batch_response(), result_response("done", pages=None)], archive
    )

    assert client.parse(pdf_file)["page_count"] == 1


def test_page_count_zero_when_no_source(pdf_file):
    archive = make_zip({"full.md": "正文"})
    client, _t, _u, _d, _c = make_client(
        [batch_response(), result_response("done", pages=None)], archive
    )

    result = client.parse(pdf_file)

    assert result["complete"] is True
    assert result["page_count"] == 0


# --------------------------------------------------------------- 失败态


def test_failed_state_returns_incomplete_without_raising(pdf_file):
    client, _t, _u, _d, _c = make_client(
        [batch_response(), result_response("failed", err_msg="page limit exceeded")]
    )

    result = client.parse(pdf_file)

    assert result["complete"] is False
    assert result["content"] == ""
    assert "page limit exceeded" in result["error"]


def test_poll_timeout_returns_incomplete(pdf_file):
    client, _t, _u, _d, _c = make_client(
        [batch_response()] + [result_response("running")] * 10,
        max_poll_seconds=10.0,
    )

    result = client.parse(pdf_file)

    assert result["complete"] is False
    assert "轮询超时" in result["error"]


def test_missing_full_md_returns_incomplete(pdf_file):
    archive = make_zip({"layout.json": "{}"})
    client, _t, _u, _d, _c = make_client(
        [batch_response(), result_response("done")], archive
    )

    result = client.parse(pdf_file)

    assert result["complete"] is False
    assert "full.md" in result["error"]


def test_empty_content_returns_incomplete(pdf_file):
    archive = make_zip({"full.md": "   \n  "})
    client, _t, _u, _d, _c = make_client(
        [batch_response(), result_response("done")], archive
    )

    result = client.parse(pdf_file)

    assert result["complete"] is False
    assert "正文为空" in result["error"]


def test_done_without_zip_url_returns_incomplete(pdf_file):
    response = result_response("done")
    response["data"]["extract_result"][0]["full_zip_url"] = ""
    client, _t, _u, _d, _c = make_client([batch_response(), response])

    result = client.parse(pdf_file)

    assert result["complete"] is False
    assert "未返回下载地址" in result["error"]


def test_download_failure_returns_incomplete(pdf_file):
    client, _t, _u, download, _c = make_client(
        [batch_response(), result_response("done")]
    )
    download.error = HttpApiError("HTTP 500", status_code=500)

    result = client.parse(pdf_file)

    assert result["complete"] is False
    assert "下载/解压" in result["error"]


def test_bad_zip_returns_incomplete(pdf_file):
    client, _t, _u, _d, _c = make_client(
        [batch_response(), result_response("done")], b"not a zip at all"
    )

    result = client.parse(pdf_file)

    assert result["complete"] is False
    assert "not a zip file" in result["error"]


def test_missing_batch_id_invalid_structure(pdf_file):
    client, _t, _u, _d, _c = make_client([{"code": 0, "data": {}}])

    result = client.parse(pdf_file)

    assert result["complete"] is False
    assert "batch_id" in result["error"]


def test_poll_invalid_structure(pdf_file):
    client, _t, _u, _d, _c = make_client([batch_response(), {"code": 0, "data": {}}])

    result = client.parse(pdf_file)

    assert result["complete"] is False
    assert "extract_result" in result["error"]


# --------------------------------------------------------------- 鉴权/重试


def test_401_not_retried(pdf_file):
    client, transport, _u, _d, _c = make_client(
        [HttpApiError("HTTP 401", status_code=401)]
    )

    result = client.parse(pdf_file)

    assert result["complete"] is False
    assert len(transport.calls) == 1  # 401 立即放弃，不重试


def test_429_retried_then_succeeds(pdf_file):
    archive = make_zip({"full.md": "正文"})
    client, transport, _u, _d, _c = make_client(
        [HttpApiError("HTTP 429", status_code=429), batch_response(), result_response("done")],
        archive,
    )

    result = client.parse(pdf_file)

    assert result["complete"] is True
    assert len(transport.calls) == 3


def test_upload_error_returns_incomplete(pdf_file):
    transcript = ScriptedJsonTransport([batch_response()])
    client = MineruClient(
        api_base_url="https://mineru.test",
        api_key="k",
        transport_json=transcript,
        upload=RecordingUpload(HttpApiError("HTTP 403", status_code=403)),
        download=StaticDownload(b""),
        sleep=lambda _s: None,
    )

    result = client.parse(pdf_file)

    assert result["complete"] is False
    assert "上传" in result["error"]


# --------------------------------------------------------------- 输入体检


def test_missing_file_returns_incomplete(tmp_path: Path):
    client, _t, _u, _d, _c = make_client([])

    result = client.parse(tmp_path / "nope.pdf")

    assert result["complete"] is False
    assert "不存在" in result["error"]


def test_oversized_file_returns_incomplete(tmp_path: Path, monkeypatch):
    path = tmp_path / "big.pdf"
    path.write_bytes(b"x")
    monkeypatch.setattr("app.models.mineru.MAX_FILE_BYTES", 0)
    client, _t, _u, _d, _c = make_client([])

    result = client.parse(path)

    assert result["complete"] is False
    assert "超过 MinerU 限制" in result["error"]


def test_reported_error_never_contains_api_key(pdf_file):
    """失败信息里不得出现密钥（错误路径也可能拼进 URL/响应）。"""
    client, _t, _u, _d, _c = make_client([batch_response(), result_response("failed", err_msg="boom")])

    result = client.parse(pdf_file)

    assert "test-key" not in result["error"]
