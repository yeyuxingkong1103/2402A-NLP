from __future__ import annotations

from io import BytesIO
from pathlib import Path, PurePosixPath
from time import monotonic, sleep
from zipfile import BadZipFile, ZipFile

import httpx

from ..config import Settings


class MinerUError(RuntimeError):
    """MinerU parsing failed and the caller should use its local fallback."""


def _api_data(response: httpx.Response) -> dict:
    response.raise_for_status()
    payload = response.json()
    if payload.get("code") != 0:
        raise MinerUError(str(payload.get("msg") or "MinerU 接口调用失败"))
    data = payload.get("data")
    if not isinstance(data, dict):
        raise MinerUError("MinerU 返回数据不完整")
    return data


def _full_markdown(zip_content: bytes) -> str:
    try:
        with ZipFile(BytesIO(zip_content)) as archive:
            names = [name for name in archive.namelist() if PurePosixPath(name).name == "full.md"]
            if not names:
                raise MinerUError("MinerU 结果中没有 full.md")
            if archive.getinfo(names[0]).file_size > 20 * 1024 * 1024:
                raise MinerUError("MinerU 结果中的 full.md 超过20MB")
            return archive.read(names[0]).decode("utf-8", errors="ignore").strip()
    except BadZipFile as exc:
        raise MinerUError("MinerU 返回的结果压缩包无效") from exc


def extract_with_mineru(path: Path, settings: Settings, force_ocr: bool = False) -> str:
    """Upload one local document to MinerU and return the parsed Markdown text."""
    if not settings.mineru_enabled or not settings.mineru_api_token:
        raise MinerUError("MinerU 未启用或未配置令牌")

    base_url = settings.mineru_api_base_url.rstrip("/")
    headers = {
        "Authorization": f"Bearer {settings.mineru_api_token}",
        "Content-Type": "application/json",
    }
    request_timeout = max(10, min(int(settings.mineru_timeout), 120))
    request_data = {
        "files": [{"name": path.name, "is_ocr": bool(force_ocr)}],
        "model_version": settings.mineru_model_version,
        "enable_table": True,
        "enable_formula": True,
        "language": "ch",
    }

    upload_data = _api_data(
        httpx.post(
            f"{base_url}/file-urls/batch",
            headers=headers,
            json=request_data,
            timeout=request_timeout,
        )
    )
    batch_id = str(upload_data.get("batch_id") or "").strip()
    file_urls = upload_data.get("file_urls") or []
    if not batch_id or not file_urls:
        raise MinerUError("MinerU 没有返回上传地址")

    with path.open("rb") as source:
        upload_response = httpx.put(str(file_urls[0]), content=source, timeout=request_timeout)
    upload_response.raise_for_status()

    deadline = monotonic() + max(1, int(settings.mineru_timeout))
    while monotonic() < deadline:
        result_data = _api_data(
            httpx.get(
                f"{base_url}/extract-results/batch/{batch_id}",
                headers=headers,
                timeout=request_timeout,
            )
        )
        results = result_data.get("extract_result") or []
        result = results[0] if results else {}
        state = str(result.get("state") or "").lower()
        if state == "failed":
            raise MinerUError(str(result.get("err_msg") or "MinerU 解析失败"))
        if state == "done":
            zip_url = str(result.get("full_zip_url") or "").strip()
            if not zip_url:
                raise MinerUError("MinerU 没有返回结果下载地址")
            download = httpx.get(zip_url, follow_redirects=True, timeout=request_timeout)
            download.raise_for_status()
            if len(download.content) > 100 * 1024 * 1024:
                raise MinerUError("MinerU 结果压缩包超过100MB")
            text = _full_markdown(download.content)
            if not text:
                raise MinerUError("MinerU 没有识别出文本")
            return text
        sleep(max(0, int(settings.mineru_poll_interval)))

    raise MinerUError("MinerU 解析超时")
