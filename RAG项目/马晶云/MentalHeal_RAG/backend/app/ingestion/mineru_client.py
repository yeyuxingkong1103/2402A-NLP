import io
import json
import time
import zipfile
from pathlib import Path
from typing import Any

import httpx


class MineruClient:
    """MinerU Cloud API v4 client for one PDF at a time."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        api_version: str = "v4",
        model_version: str = "pipeline",
        poll_seconds: int = 5,
        timeout_seconds: int = 900,
        is_ocr: bool = True,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.api_version = api_version
        self.model_version = model_version
        self.poll_seconds = poll_seconds
        self.timeout_seconds = timeout_seconds
        self.is_ocr = is_ocr

    def parse(self, path: Path, image_dir: Path | None = None) -> dict[int, str]:
        if not self.api_key:
            raise ValueError("MINERU_API_KEY is required")
        return self._parse_part(path, image_dir, 0)

    def _parse_part(
        self,
        path: Path,
        image_dir: Path | None,
        page_offset: int,
    ) -> dict[int, str]:
        with httpx.Client(timeout=60.0, follow_redirects=True) as client:
            upload = self._request_upload_url(client, path)
            file_url = self._upload_url(upload)
            if not file_url:
                raise RuntimeError(f"MinerU did not return an upload URL: {upload}")
            batch_id = self._batch_id(upload)
            if not batch_id:
                raise RuntimeError(f"MinerU did not return a batch ID: {upload}")
            self._upload_file(client, file_url, path)
            result = self._poll_result(client, batch_id, path.name)
            pages = self._download_markdown(client, result, image_dir)
        return {page_number + page_offset: text for page_number, text in pages.items()}

    def _request_upload_url(self, client: httpx.Client, path: Path) -> dict[str, Any]:
        url = f"{self.base_url}/{self.api_version}/file-urls/batch"
        payload = {
            "files": [{"name": path.name, "is_ocr": self.is_ocr}],
            "model_version": self.model_version,
        }
        response = client.post(url, headers=self._headers(), json=payload)
        response.raise_for_status()
        return response.json()

    def _upload_file(self, client: httpx.Client, upload_url: str, path: Path) -> None:
        response = client.put(upload_url, content=path.read_bytes())
        response.raise_for_status()

    def _poll_result(
        self,
        client: httpx.Client,
        batch_id: str,
        filename: str,
    ) -> dict[str, Any]:
        url = f"{self.base_url}/{self.api_version}/extract-results/batch/{batch_id}"
        deadline = time.monotonic() + self.timeout_seconds
        while time.monotonic() < deadline:
            try:
                response = client.get(url, headers=self._headers())
                response.raise_for_status()
                payload = response.json()
            except (httpx.HTTPError, ValueError):
                payload = self._retry_poll_request(client, url)
                if payload is None:
                    time.sleep(self.poll_seconds)
                    continue
            result = self._find_result(payload, filename)
            state = str(result.get("state", "")).lower()
            if state in {"done", "success", "succeeded"}:
                return result
            if state in {"failed", "error"}:
                raise RuntimeError(f"MinerU task failed: {result}")
            time.sleep(self.poll_seconds)
        raise TimeoutError(f"MinerU task timed out after {self.timeout_seconds} seconds")

    def _retry_poll_request(self, client: httpx.Client, url: str) -> dict[str, Any] | None:
        for _ in range(2):
            try:
                response = client.get(url, headers=self._headers(), timeout=120.0)
                response.raise_for_status()
                return response.json()
            except (httpx.HTTPError, ValueError):
                time.sleep(2)
        try:
            with httpx.Client(timeout=120.0, follow_redirects=True, trust_env=False) as direct_client:
                response = direct_client.get(url, headers=self._headers())
                response.raise_for_status()
                return response.json()
        except (httpx.HTTPError, ValueError):
            return None

    def _download_markdown(
        self,
        client: httpx.Client,
        result: dict[str, Any],
        image_dir: Path | None,
    ) -> dict[int, str]:
        result_url = result.get("full_zip_url") or result.get("fullZipUrl")
        if not result_url:
            inline = result.get("markdown") or result.get("content")
            return self._split_pages(str(inline or ""))
        response = self._get_result_package(client, str(result_url))
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            if image_dir:
                self._extract_images(archive, image_dir)
            page_text = self._content_list_pages(archive)
            if page_text:
                return page_text
            markdown_files = [name for name in archive.namelist() if name.lower().endswith(".md")]
            text = "\n\n".join(
                archive.read(name).decode("utf-8", errors="replace") for name in markdown_files
            )
        return self._split_pages(text)

    def _get_result_package(self, client: httpx.Client, result_url: str) -> httpx.Response:
        last_error: Exception | None = None
        for _ in range(3):
            try:
                response = client.get(result_url, timeout=120.0)
                response.raise_for_status()
                return response
            except httpx.HTTPError as error:
                last_error = error
                time.sleep(2)
        with httpx.Client(timeout=120.0, follow_redirects=True, trust_env=False) as direct_client:
            response = direct_client.get(result_url)
            response.raise_for_status()
            return response

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    @staticmethod
    def _batch_id(payload: dict[str, Any]) -> str:
        data = payload.get("data", payload)
        return str(data.get("batch_id") or data.get("batchId") or "")

    @staticmethod
    def _upload_url(payload: dict[str, Any]) -> str:
        data = payload.get("data", payload)
        urls = data.get("file_urls", data.get("fileUrls", [])) if isinstance(data, dict) else []
        if isinstance(urls, list) and urls:
            first = urls[0]
            return str(first.get("url", first) if isinstance(first, dict) else first)
        return ""

    @staticmethod
    def _find_result(payload: dict[str, Any], filename: str) -> dict[str, Any]:
        data = payload.get("data", payload)
        results = data.get("extract_result", data.get("extract_results", []))
        if isinstance(results, dict):
            results = [results]
        if not isinstance(results, list) or not results:
            raise RuntimeError(f"MinerU returned no extraction result: {json.dumps(payload, ensure_ascii=False)}")
        for result in results:
            if str(result.get("file_name", result.get("data_id", ""))) in {filename, Path(filename).stem}:
                return result
        return results[0]

    @staticmethod
    def _extract_images(archive: zipfile.ZipFile, image_dir: Path) -> None:
        image_dir.mkdir(parents=True, exist_ok=True)
        allowed = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
        for index, name in enumerate(archive.namelist(), start=1):
            source = Path(name)
            if source.suffix.lower() not in allowed:
                continue
            target = image_dir / f"image_{index:04d}{source.suffix.lower()}"
            target.write_bytes(archive.read(name))

    @staticmethod
    def _content_list_pages(archive: zipfile.ZipFile) -> dict[int, str]:
        content_name = next(
            (
                name
                for name in archive.namelist()
                if Path(name).name == "content_list.json"
                or Path(name).name.endswith("_content_list.json")
            ),
            None,
        )
        if not content_name:
            return {}
        try:
            payload = json.loads(archive.read(content_name).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return {}
        if not isinstance(payload, list):
            return {}
        pages: dict[int, list[str]] = {}
        for item in payload:
            if not isinstance(item, dict) or not isinstance(item.get("page_idx"), int):
                continue
            text = item.get("text") or item.get("table_body") or item.get("content") or ""
            if not isinstance(text, str) or not text.strip():
                continue
            page_number = item["page_idx"] + 1
            pages.setdefault(page_number, []).append(text.strip())
        return {page: "\n\n".join(texts) for page, texts in pages.items()}

    @staticmethod
    def _split_pages(markdown: str) -> dict[int, str]:
        if not markdown.strip():
            return {}
        pages = markdown.split("\n---\n")
        return {index: page.strip() for index, page in enumerate(pages, start=1) if page.strip()}
