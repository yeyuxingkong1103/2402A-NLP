from dataclasses import replace
from io import BytesIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import httpx
import pytest

from backend.app.config import Settings
from backend.app.workspace import file_utits
from backend.app.workspace.mineru import MinerUError, _full_markdown, extract_with_mineru


def mineru_settings() -> Settings:
    return replace(
        Settings(),
        mineru_enabled=True,
        mineru_api_token="secret-token",
        mineru_api_base_url="https://mineru.example/api/v4",
        mineru_model_version="vlm",
        mineru_timeout=30,
        mineru_poll_interval=0,
    )


def response(url: str, payload: dict | None = None, content: bytes = b"") -> httpx.Response:
    return httpx.Response(
        200,
        request=httpx.Request("GET", url),
        json=payload if payload is not None else None,
        content=content if payload is None else None,
    )


def result_zip(markdown: str, name: str = "result/full.md") -> bytes:
    target = BytesIO()
    with ZipFile(target, "w") as archive:
        archive.writestr(name, markdown)
    return target.getvalue()


def test_mineru_uploads_polls_and_reads_full_markdown_without_leaking_token(tmp_path, monkeypatch):
    path = tmp_path / "合同.pdf"
    path.write_bytes(b"pdf-content")
    calls: list[tuple[str, str]] = []

    def post(url, *, headers, json, timeout):
        calls.append(("post", url))
        assert headers["Authorization"] == "Bearer secret-token"
        assert json["model_version"] == "vlm"
        assert json["files"] == [{"name": "合同.pdf", "is_ocr": True}]
        return response(url, {"code": 0, "data": {"batch_id": "batch-1", "file_urls": ["https://upload.example/signed"]}})

    def put(url, *, content, timeout):
        calls.append(("put", url))
        assert content.read() == b"pdf-content"
        return response(url, {"ok": True})

    def get(url, **kwargs):
        calls.append(("get", url))
        if "extract-results" in url:
            assert kwargs["headers"]["Authorization"] == "Bearer secret-token"
            return response(url, {"code": 0, "data": {"extract_result": [{"state": "done", "full_zip_url": "https://cdn.example/result.zip"}]}})
        assert "headers" not in kwargs
        return response(url, content=result_zip("# 合同\n租金为1000元"))

    monkeypatch.setattr("backend.app.workspace.mineru.httpx.post", post)
    monkeypatch.setattr("backend.app.workspace.mineru.httpx.put", put)
    monkeypatch.setattr("backend.app.workspace.mineru.httpx.get", get)

    text = extract_with_mineru(path, mineru_settings(), force_ocr=True)

    assert text == "# 合同\n租金为1000元"
    assert calls == [
        ("post", "https://mineru.example/api/v4/file-urls/batch"),
        ("put", "https://upload.example/signed"),
        ("get", "https://mineru.example/api/v4/extract-results/batch/batch-1"),
        ("get", "https://cdn.example/result.zip"),
    ]


def test_mineru_rejects_result_zip_without_full_markdown(tmp_path, monkeypatch):
    path = tmp_path / "scan.pdf"
    path.write_bytes(b"pdf")

    monkeypatch.setattr(
        "backend.app.workspace.mineru.httpx.post",
        lambda url, **kwargs: response(url, {"code": 0, "data": {"batch_id": "batch-1", "file_urls": ["https://upload.example/signed"]}}),
    )
    monkeypatch.setattr(
        "backend.app.workspace.mineru.httpx.put",
        lambda url, **kwargs: response(url, {"ok": True}),
    )

    def get(url, **kwargs):
        if "extract-results" in url:
            return response(url, {"code": 0, "data": {"extract_result": [{"state": "done", "full_zip_url": "https://cdn.example/result.zip"}]}})
        return response(url, content=result_zip("not markdown", "result/content.json"))

    monkeypatch.setattr("backend.app.workspace.mineru.httpx.get", get)

    with pytest.raises(MinerUError, match="full.md"):
        extract_with_mineru(path, mineru_settings())


def test_mineru_reports_failed_task(tmp_path, monkeypatch):
    path = tmp_path / "scan.pdf"
    path.write_bytes(b"pdf")

    monkeypatch.setattr(
        "backend.app.workspace.mineru.httpx.post",
        lambda url, **kwargs: response(url, {"code": 0, "data": {"batch_id": "batch-1", "file_urls": ["https://upload.example/signed"]}}),
    )
    monkeypatch.setattr(
        "backend.app.workspace.mineru.httpx.put",
        lambda url, **kwargs: response(url, {"ok": True}),
    )
    monkeypatch.setattr(
        "backend.app.workspace.mineru.httpx.get",
        lambda url, **kwargs: response(url, {"code": 0, "data": {"extract_result": [{"state": "failed", "err_msg": "文件损坏"}]}}),
    )

    with pytest.raises(MinerUError, match="文件损坏"):
        extract_with_mineru(path, mineru_settings())


class NoMultimodalCall:
    def analyze_media(self, *args, **kwargs):
        raise AssertionError("不应调用原多模态模型")


def test_image_uses_mineru_before_local_ocr_and_multimodal(tmp_path, monkeypatch):
    path = tmp_path / "证据.png"
    path.write_bytes(b"image")
    monkeypatch.setattr(file_utits, "extract_with_mineru", lambda *args, **kwargs: "MinerU识别结果", raising=False)

    text, metadata = file_utits.extract_text(path, path.name, mineru_settings(), NoMultimodalCall())

    assert text == "MinerU识别结果"
    assert metadata["extraction_method"] == "mineru"
    assert metadata["mineru_status"] == "success"


def test_low_text_pdf_uses_mineru_with_ocr(tmp_path, monkeypatch):
    path = tmp_path / "扫描件.pdf"
    path.write_bytes(b"pdf")
    received = {}
    monkeypatch.setattr(file_utits, "parse_text", lambda *args: "")
    monkeypatch.setattr(file_utits, "pdf_has_tables", lambda *args: False, raising=False)

    def mineru(path, settings, force_ocr=False):
        received["force_ocr"] = force_ocr
        return "扫描件完整内容"

    monkeypatch.setattr(file_utits, "extract_with_mineru", mineru, raising=False)

    text, metadata = file_utits.extract_text(path, path.name, mineru_settings(), NoMultimodalCall())

    assert text == "扫描件完整内容"
    assert received["force_ocr"] is True
    assert metadata["extraction_method"] == "mineru"


def test_pdf_with_table_uses_mineru_without_forced_ocr(tmp_path, monkeypatch):
    path = tmp_path / "费用表.pdf"
    path.write_bytes(b"pdf")
    local_text = "本地可读取文字" * 20
    received = {}
    monkeypatch.setattr(file_utits, "parse_text", lambda *args: local_text)
    monkeypatch.setattr(file_utits, "pdf_has_tables", lambda *args: True, raising=False)

    def mineru(path, settings, force_ocr=False):
        received["force_ocr"] = force_ocr
        return "|项目|金额|\n|租金|1000元|"

    monkeypatch.setattr(file_utits, "extract_with_mineru", mineru, raising=False)

    text, metadata = file_utits.extract_text(path, path.name, mineru_settings(), NoMultimodalCall())

    assert text.startswith("|项目|金额|")
    assert received["force_ocr"] is False
    assert metadata["extraction_method"] == "mineru"


def test_simple_text_pdf_keeps_fast_local_parser(tmp_path, monkeypatch):
    path = tmp_path / "普通判决书.pdf"
    path.write_bytes(b"pdf")
    local_text = "普通正文内容" * 30
    monkeypatch.setattr(file_utits, "parse_text", lambda *args: local_text)
    monkeypatch.setattr(file_utits, "pdf_has_tables", lambda *args: False, raising=False)
    monkeypatch.setattr(file_utits, "extract_with_mineru", lambda *args, **kwargs: pytest.fail("简单PDF不应调用MinerU"), raising=False)

    text, metadata = file_utits.extract_text(path, path.name, mineru_settings(), NoMultimodalCall())

    assert text == local_text
    assert metadata["extraction_method"] == "text_parser"
    assert metadata["mineru_status"] == "not_used"


def test_mineru_failure_falls_back_to_existing_image_flow(tmp_path, monkeypatch):
    path = tmp_path / "聊天记录.png"
    path.write_bytes(b"image")
    monkeypatch.setattr(file_utits, "extract_with_mineru", lambda *args, **kwargs: (_ for _ in ()).throw(MinerUError("temporary failure")), raising=False)
    monkeypatch.setattr(file_utits, "parse_text", lambda *args: "OCR备用结果")

    class EmptyMultimodal:
        def analyze_media(self, *args, **kwargs):
            return ""

    text, metadata = file_utits.extract_text(path, path.name, mineru_settings(), EmptyMultimodal())

    assert text == "OCR备用结果"
    assert metadata["extraction_method"] == "ocr"
    assert metadata["mineru_status"] == "failed"


def test_mineru_fallback_log_does_not_include_signed_url(tmp_path, monkeypatch, caplog):
    path = tmp_path / "聊天记录.png"
    path.write_bytes(b"image")
    signed_url = "https://upload.example/signed?secret=do-not-log"
    monkeypatch.setattr(file_utits, "extract_with_mineru", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError(signed_url)))
    monkeypatch.setattr(file_utits, "parse_text", lambda *args: "OCR备用结果")

    class EmptyMultimodal:
        def analyze_media(self, *args, **kwargs):
            return ""

    with caplog.at_level("WARNING"):
        text, metadata = file_utits.extract_text(path, path.name, mineru_settings(), EmptyMultimodal())

    assert text == "OCR备用结果"
    assert metadata["mineru_status"] == "failed"
    logged = "\n".join(
        [record.getMessage() + str(getattr(record, "fields", "")) for record in caplog.records]
    )
    assert signed_url not in logged


def test_full_markdown_rejects_oversized_uncompressed_content():
    target = BytesIO()
    with ZipFile(target, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("result/full.md", b"x" * (20 * 1024 * 1024 + 1))

    with pytest.raises(MinerUError, match="full.md.*20MB"):
        _full_markdown(target.getvalue())
