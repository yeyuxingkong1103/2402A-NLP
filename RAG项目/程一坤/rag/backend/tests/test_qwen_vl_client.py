"""批次 22：Qwen-VL 兜底客户端单测（app/models/qwen_vl.py）。

全部用替身渲染器与替身传输，不联网、不依赖 PyMuPDF。覆盖：
- 正常识别、多页拼接
- 分批：max_pages_per_request 决定批次数与请求次数
- 兼容模式返回结构（string / 数组形态）
- 401 不重试、429 重试、结构异常
- 渲染失败（依赖缺失 / 渲染报错 / 零页）→ 返回空串 + last_error
- 失败信息不含密钥
"""

import base64
import json
from pathlib import Path

import pytest

from app.models.http_retry import HttpApiError
from app.models.qwen_vl import QwenVlClient

FAKE_PNG = b"\x89PNG\r\n\x1a\nfake-page"


class ScriptedTransport:
    """按脚本返回响应；记录每次调用的 url / headers / payload。"""

    def __init__(self, script: list) -> None:
        self.script = list(script)
        self.calls: list[tuple[str, dict, bytes]] = []

    def __call__(self, url, headers, payload, timeout):
        self.calls.append((url, dict(headers), payload))
        action = self.script.pop(0)
        if isinstance(action, Exception):
            raise action
        return action


class FakeRenderer:
    """返回构造好的页图字节；可注入异常。"""

    def __init__(self, pages: int = 1, error: Exception | None = None) -> None:
        self.pages = pages
        self.error = error
        self.calls = 0

    def __call__(self, source_path: Path) -> list[bytes]:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return [FAKE_PNG] * self.pages


def ocr_response(text: str) -> dict:
    return {"choices": [{"message": {"content": text}}]}


@pytest.fixture()
def pdf_file(tmp_path: Path) -> Path:
    path = tmp_path / "scan.pdf"
    path.write_bytes(b"%PDF-1.4 fake")
    return path


def make_client(script: list, pages: int = 1, **kwargs):
    transport = ScriptedTransport(script)
    renderer = FakeRenderer(pages)
    client = QwenVlClient(
        api_base_url="https://dashscope.test/compatible-mode/v1",
        api_key="test-key",
        model="qwen-vl-ocr",
        renderer=renderer,
        transport=transport,
        sleep=lambda _s: None,
        monotonic=lambda: 0.0,
        **kwargs,
    )
    return client, transport, renderer


# --------------------------------------------------------------- 正常路径


def test_parse_single_page(pdf_file):
    client, transport, renderer = make_client([ocr_response("第一章 总则\n第一条 正文")], pages=1)

    text = client.parse(pdf_file)

    assert "第一条 正文" in text
    assert renderer.calls == 1
    assert client.last_trace["page_count"] == 1
    _url, headers, payload = transport.calls[0]
    assert _url.endswith("/chat/completions")
    assert headers["Authorization"] == "Bearer test-key"
    body = json.loads(payload.decode("utf-8"))
    assert body["model"] == "qwen-vl-ocr"
    # 图片以 data URL 承载
    parts = body["messages"][0]["content"]
    assert parts[0]["type"] == "image_url"
    assert parts[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert parts[0]["min_pixels"] == 3072
    assert parts[0]["max_pixels"] == 8388608
    assert parts[-1]["type"] == "text"


def test_parse_joins_multiple_batches(pdf_file):
    client, transport, _renderer = make_client(
        [ocr_response("第1页"), ocr_response("第2页"), ocr_response("第3页")],
        pages=5,
        max_pages_per_request=2,
    )

    text = client.parse(pdf_file)

    assert text == "第1页\n第2页\n第3页"
    assert len(transport.calls) == 3  # 5 页按每批 2 页 → 3 批
    counts = [b["images"] for b in client.last_trace["batches"]]
    assert counts == [2, 2, 1]


def test_parse_single_batch_when_pages_within_limit(pdf_file):
    client, transport, _renderer = make_client([ocr_response("全部文字")], pages=3)

    text = client.parse(pdf_file)

    assert text == "全部文字"
    assert len(transport.calls) == 1


# --------------------------------------------------- 分批（体积才是真正约束）


def _placeholder(tmp_path: Path) -> Path:
    """体积分批用例只需一个存在的文件（渲染器是替身，不真读内容）。"""
    path = tmp_path / "placeholder.pdf"
    path.write_bytes(b"%PDF-1.4")
    return path


class SizedRenderer:
    """每页返回指定字节数的假 PNG。"""

    def __init__(self, page_bytes: int, pages: int) -> None:
        self.page_bytes = page_bytes
        self.pages = pages

    def __call__(self, source_path: Path) -> list[bytes]:
        return [FAKE_PNG + b"\x00" * (self.page_bytes - len(FAKE_PNG))] * self.pages


def test_batches_limited_by_payload_size(tmp_path):
    """★ 请求体超约 2MB 时 qwen-vl-ocr 只回第一页 -> 必须按体积分批。"""
    transport = ScriptedTransport([ocr_response(f"第{i}批") for i in range(10)])
    # 每页 300KB 原始字节 → base64 后约 400KB/页 → 1MB 上限下每批最多 2 页
    client = QwenVlClient(
        api_base_url="https://dashscope.test/v1",
        api_key="k",
        renderer=SizedRenderer(page_bytes=300_000, pages=7),
        transport=transport,
        max_pages_per_request=8,
        max_request_bytes=1_000_000,
        sleep=lambda _s: None,
        monotonic=lambda: 0.0,
    )

    client.parse(_placeholder(tmp_path))

    counts = [b["images"] for b in client.last_trace["batches"]]
    assert counts == [2, 2, 2, 1], counts
    for batch in client.last_trace["batches"]:
        assert batch["payload_bytes"] <= 1_000_000


def test_batches_limited_by_page_count_when_images_are_small(pdf_file):
    """页图很小时体积约束不触发，退化为按页数上限分批。"""
    client, transport, _renderer = make_client(
        [ocr_response("a"), ocr_response("b"), ocr_response("c")],
        pages=6,
        max_pages_per_request=2,
        max_request_bytes=100_000_000,
    )

    client.parse(pdf_file)

    assert [b["images"] for b in client.last_trace["batches"]] == [2, 2, 2]
    assert len(transport.calls) == 3


def test_single_oversized_page_sent_alone(tmp_path):
    """单页本身就超上限时不能丢页：单独成批并记警告。"""
    transport = ScriptedTransport([ocr_response("大字页")])
    client = QwenVlClient(
        api_base_url="https://dashscope.test/v1",
        api_key="k",
        renderer=SizedRenderer(page_bytes=100_000, pages=1),
        transport=transport,
        max_pages_per_request=8,
        max_request_bytes=1000,  # 故意设得比单页还小
        sleep=lambda _s: None,
        monotonic=lambda: 0.0,
    )

    assert client.parse(_placeholder(tmp_path)) == "大字页"
    assert len(transport.calls) == 1
    assert [b["images"] for b in client.last_trace["batches"]] == [1]


def test_default_batch_plan_for_typical_pages(tmp_path):
    """150 DPI 中文法规页约 290KB（base64 后约 400KB）→ 1.5MB 上限下每批 3 页。"""
    transport = ScriptedTransport([ocr_response(f"p{i}") for i in range(10)])
    client = QwenVlClient(
        api_base_url="https://dashscope.test/v1",
        api_key="k",
        renderer=SizedRenderer(page_bytes=294_000, pages=10),
        transport=transport,
        sleep=lambda _s: None,
        monotonic=lambda: 0.0,
    )

    client.parse(_placeholder(tmp_path))

    counts = [b["images"] for b in client.last_trace["batches"]]
    assert counts == [3, 3, 3, 1], counts


def test_content_as_parts_list_is_concatenated(pdf_file):
    response = {"choices": [{"message": {"content": [{"type": "text", "text": "甲"}, {"type": "text", "text": "乙"}]}}]}
    client, _t, _r = make_client([response], pages=1)

    assert client.parse(pdf_file) == "甲乙"


def test_blank_batch_output_is_skipped(pdf_file):
    client, _t, _r = make_client(
        [ocr_response("   "), ocr_response("有字")], pages=2, max_pages_per_request=1
    )

    assert client.parse(pdf_file) == "有字"


# --------------------------------------------------------------- 失败态


def test_401_returns_empty_and_not_retried(pdf_file):
    client, transport, _r = make_client(
        [HttpApiError("HTTP 401", status_code=401)], pages=1
    )

    text = client.parse(pdf_file)

    assert text == ""
    assert len(transport.calls) == 1
    assert "401" in client.last_error


def test_429_retried_then_succeeds(pdf_file):
    client, transport, _r = make_client(
        [HttpApiError("HTTP 429", status_code=429), ocr_response("重试成功")], pages=1
    )

    assert client.parse(pdf_file) == "重试成功"
    assert len(transport.calls) == 2


def test_timeout_exhausts_retries_and_returns_empty(pdf_file):
    client, transport, _r = make_client([TimeoutError("timed out")] * 4, pages=1)

    text = client.parse(pdf_file)

    assert text == ""
    assert len(transport.calls) == 4  # 1 次首调 + 3 次重试后放弃
    assert "调用失败" in client.last_error
    assert "TimeoutError" in client.last_error


def test_invalid_response_structure_returns_empty(pdf_file):
    client, _t, _r = make_client([{"nope": 1}], pages=1)

    text = client.parse(pdf_file)

    assert text == ""
    assert "结构无效" in client.last_error


def test_all_blank_output_reports_no_text(pdf_file):
    client, _t, _r = make_client([ocr_response("   ")], pages=1)

    text = client.parse(pdf_file)

    assert text == ""
    assert "未识别出任何文字" in client.last_error


def test_missing_file_returns_empty(tmp_path: Path):
    client, _t, renderer = make_client([], pages=1)

    text = client.parse(tmp_path / "nope.pdf")

    assert text == ""
    assert "不存在" in client.last_error
    assert renderer.calls == 0


def test_renderer_import_error_mentions_pymupdf(pdf_file: Path):
    client, _t, _r = make_client([], pages=1)
    client.renderer = FakeRenderer(error=ImportError("需要 PyMuPDF 渲染页图"))

    text = client.parse(pdf_file)

    assert text == ""
    assert "渲染依赖" in client.last_error


def test_renderer_failure_returns_empty(pdf_file):
    client, _t, _r = make_client([], pages=1)
    client.renderer = FakeRenderer(error=ValueError("corrupt pdf"))

    text = client.parse(pdf_file)

    assert text == ""
    assert "渲染失败" in client.last_error


def test_zero_pages_returns_empty(pdf_file):
    client, _t, _r = make_client([], pages=0)

    text = client.parse(pdf_file)

    assert text == ""
    assert "未渲染出任何页面" in client.last_error


def test_error_never_contains_api_key(pdf_file):
    client, _t, _r = make_client(
        [HttpApiError("HTTP 500", status_code=500)] * 4, pages=1
    )

    client.parse(pdf_file)

    assert client.last_error
    assert "test-key" not in client.last_error


def test_default_renderer_is_used_when_not_injected():
    client = QwenVlClient(api_base_url="u", api_key="k", transport=lambda *a: {})
    assert client.renderer == client._render_pages


def test_renderer_produces_png_bytes_for_real_pdf():
    """默认渲染器接一次真实小 PDF：断言输出是 PNG 字节（依赖 PyMuPDF）。"""
    fitz = pytest.importorskip("fitz")
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "Hello PDF")
    payload = document.tobytes()
    document.close()

    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "mini.pdf"
        path.write_bytes(payload)
        client = QwenVlClient(api_base_url="u", api_key="k", dpi=72)
        images = client.renderer(path)

    assert len(images) == 1
    assert images[0][:8] == b"\x89PNG\r\n\x1a\n"


def test_payload_base64_roundtrip(pdf_file):
    """发出的图片 data URL 必须能被解回原始 PNG 字节。"""
    client, transport, _r = make_client([ocr_response("x")], pages=1)

    client.parse(pdf_file)

    body = json.loads(transport.calls[0][2].decode("utf-8"))
    url = body["messages"][0]["content"][0]["image_url"]["url"]
    encoded = url.split(",", 1)[1]
    assert base64.b64decode(encoded) == FAKE_PNG
