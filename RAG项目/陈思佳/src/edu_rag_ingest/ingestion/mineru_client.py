from __future__ import annotations

"""MinerU API 客户端，负责上传文件、轮询解析结果并提取 Markdown。"""

import io
import json
import logging
import time
import zipfile
from pathlib import Path
from typing import Any

import httpx

from ..config.config import MinerUConfig

logger = logging.getLogger(__name__)


class MinerUClient:
    def __init__(self, config: MinerUConfig) -> None:
        self.config = config
        self.config.parsed_output_dir.mkdir(parents=True, exist_ok=True)
        self.config.artifact_output_dir.mkdir(parents=True, exist_ok=True)
        self.base_url = config.api_url.rstrip("/")

    def parse_file(self, file_path: Path) -> str:
        if not self.config.enabled:
            raise RuntimeError("MinerU 解析已在配置中关闭")
        if not self.base_url:
            raise ValueError("请在配置中设置 mineru.api_url")
        if not self.config.api_key:
            raise ValueError("请设置环境变量 MINERU_API_KEY")

        logger.info("调用 MinerU 精准解析：%s", file_path)
        batch_id, upload_url = self._request_upload_url(file_path)
        self._upload_file(upload_url, file_path)
        full_zip_url = self._wait_batch_done(batch_id)
        archive_bytes = self._download_archive(full_zip_url)
        return self._save_artifacts_from_zip(file_path, archive_bytes)

    def parse_or_read(self, file_path: Path) -> str:
        suffix = file_path.suffix.lower()
        if suffix in {".md", ".txt"}:
            return file_path.read_text(encoding="utf-8", errors="ignore")
        return self.parse_file(file_path)

    def save_parsed_text(self, source_path: Path, text: str) -> Path:
        output_path = self.config.parsed_output_dir / f"{source_path.stem}.md"
        output_path.write_text(text, encoding="utf-8")
        return output_path

    def _request_upload_url(self, file_path: Path) -> tuple[str, str]:
        payload = {
            "files": [
                {
                    "name": file_path.name,
                    "data_id": file_path.stem,
                    **self.config.extra_form,
                }
            ]
        }
        response = httpx.post(
            f"{self.base_url}/api/v4/file-urls/batch",
            headers=self._json_headers(),
            json=payload,
            timeout=self.config.timeout_seconds,
        )
        response.raise_for_status()
        data = self._require_success(response.json())
        batch_id = data.get("batch_id")
        file_urls = data.get("file_urls") or []
        if not batch_id or not file_urls:
            raise RuntimeError(f"MinerU 上传申请响应异常：{json.dumps(data, ensure_ascii=False)}")
        return str(batch_id), str(file_urls[0])

    def _upload_file(self, upload_url: str, file_path: Path) -> None:
        with file_path.open("rb") as file:
            response = httpx.put(upload_url, content=file.read(), timeout=self.config.timeout_seconds)
        response.raise_for_status()

    def _wait_batch_done(self, batch_id: str) -> str:
        deadline = time.time() + self.config.timeout_seconds
        result_url = f"{self.base_url}/api/v4/extract-results/batch/{batch_id}"

        while time.time() < deadline:
            response = httpx.get(result_url, headers=self._auth_headers(), timeout=30)
            response.raise_for_status()
            payload = response.json()
            data = self._require_success(payload)
            full_zip_url = self._find_first_done_zip_url(data)
            if full_zip_url:
                return full_zip_url
            failed_message = self._find_failed_message(data)
            if failed_message:
                raise RuntimeError(f"MinerU 解析失败：{failed_message}")
            logger.info("MinerU 任务处理中，batch_id=%s", batch_id)
            time.sleep(5)

        raise TimeoutError(f"MinerU 解析超时：batch_id={batch_id}")

    def _download_archive(self, full_zip_url: str) -> bytes:
        response = httpx.get(full_zip_url, timeout=self.config.timeout_seconds)
        response.raise_for_status()
        return response.content

    def _save_artifacts_from_zip(self, source_path: Path, archive_bytes: bytes) -> str:
        archive_path = self.config.artifact_output_dir / f"{source_path.stem}.zip"
        extract_dir = self.config.artifact_output_dir / source_path.stem
        archive_path.write_bytes(archive_bytes)
        extract_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            self._extract_archive_safely(archive, extract_dir)
            markdown_names = [name for name in archive.namelist() if name.lower().endswith(".md")]
            if not markdown_names:
                raise RuntimeError("MinerU 结果压缩包中未找到 Markdown 文件")
            markdown_names.sort(key=len)
            with archive.open(markdown_names[0]) as file:
                markdown = file.read().decode("utf-8", errors="ignore")

        return markdown

    @staticmethod
    def _extract_archive_safely(archive: zipfile.ZipFile, output_dir: Path) -> None:
        output_root = output_dir.resolve()
        for member in archive.infolist():
            member_path = output_dir / member.filename
            target_path = member_path.resolve()
            if output_root != target_path and output_root not in target_path.parents:
                raise RuntimeError(f"MinerU 结果压缩包包含不安全路径：{member.filename}")
            if member.is_dir():
                target_path.mkdir(parents=True, exist_ok=True)
                continue
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target_path.open("wb") as target:
                target.write(source.read())

    def _auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.config.api_key}"}

    def _json_headers(self) -> dict[str, str]:
        return {**self._auth_headers(), "Content-Type": "application/json"}

    @staticmethod
    def _require_success(payload: dict[str, Any]) -> dict[str, Any]:
        code = payload.get("code")
        if code not in (0, "0", 200, "200", None):
            raise RuntimeError(f"MinerU API 返回错误：{json.dumps(payload, ensure_ascii=False)}")
        data = payload.get("data")
        if not isinstance(data, dict):
            raise RuntimeError(f"MinerU API 响应缺少 data：{json.dumps(payload, ensure_ascii=False)}")
        return data

    @staticmethod
    def _find_first_done_zip_url(data: dict[str, Any]) -> str | None:
        candidates: list[dict[str, Any]] = []
        for value in data.values():
            if isinstance(value, list):
                candidates.extend(item for item in value if isinstance(item, dict))
        if not candidates and any(key in data for key in ("state", "full_zip_url")):
            candidates.append(data)

        for item in candidates:
            if item.get("state") == "done" and item.get("full_zip_url"):
                return str(item["full_zip_url"])
        return None

    @staticmethod
    def _find_failed_message(data: dict[str, Any]) -> str | None:
        candidates: list[dict[str, Any]] = []
        for value in data.values():
            if isinstance(value, list):
                candidates.extend(item for item in value if isinstance(item, dict))
        if not candidates and any(key in data for key in ("state", "err_msg")):
            candidates.append(data)

        for item in candidates:
            if item.get("state") == "failed":
                return str(item.get("err_msg") or item)
        return None
