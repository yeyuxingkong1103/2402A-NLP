"""MinerU 上传与轮询下载流程：四步调用链的编排。

从 `app.models.mineru` 按职责拆出（批次 25-2）。覆盖 MinerU v4 的四步：

    1) `_create_batch`   POST file-urls/batch，拿 batch_id 与预签名上传地址
    2) `_upload_file`    PUT 原始字节（★ 绝不携带任何请求头，见方法 docstring）
    3) `_await_result`   GET extract-results/batch/{id} 轮询到 done/failed/超时
    4) `_download_zip`   GET 预签名地址取结果包（同为"不带 Authorization"）

分工：
- 请求怎么发（鉴权头 / 重试 / 传输实现）→ `app/models/mineru_api.py`
- 结果包怎么解（正文 / 页数）           → `app/models/mineru_result.py`
- 对外入口 `parse()` 与"本地先体检"的 `_build_payload()` 留在 `app/models/mineru.py`
  （前者是编排入口、后者依赖模块级常量 `MAX_FILE_BYTES`，而该常量被单测
   monkeypatch 直接替换，必须与调用点处于同一模块命名空间）。

★ 兼容性提醒：本模块方法名与签名保持原样，方法体逐字未改。
"""

from __future__ import annotations

import io
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from app.models.http_retry import HttpApiError

# 轮询时视为「任务已结束」的状态
_TERMINAL_STATES = {"done", "failed"}


class MineruFlowMixin:
    """MinerU 第 1~4 步的流程方法集合（供 `MineruClient` 组合）。

    本类不定义 `__init__`，依赖宿主类提供：
    `api_base_url` / `timeout` / `consumer`-side 的 `transport_json`、`upload`、
    `download` 三个传输点，以及轮询参数 `poll_interval_seconds` /
    `max_poll_seconds` 与 `sleep` / `monotonic`；另需 `_auth_headers()`、
    `_call()`（来自 `MineruApiMixin`）。
    """

    # 宿主类注入的属性（纯注解，不赋默认值）
    api_base_url: str
    timeout: float
    poll_interval_seconds: float
    max_poll_seconds: float
    monotonic: Callable[[], float]
    sleep: Callable[[float], None]

    def _create_batch(self, payload: bytes, trace: dict[str, Any]) -> tuple[str, str]:
        """第 1 步：申请批量上传地址，拿 batch_id 与预签名 URL。"""
        url = f"{self.api_base_url}/api/v4/file-urls/batch"
        headers = self._auth_headers()
        response = self._call(
            "MinerU 提交任务",
            lambda: self.transport_json(url, headers, payload, self.timeout),
        )
        data = response.get("data") if isinstance(response, dict) else None
        if not isinstance(data, dict):
            raise HttpApiError("MinerU 返回结构无效：缺少 data")
        batch_id = data.get("batch_id")
        file_urls = data.get("file_urls")
        if not batch_id or not isinstance(file_urls, list) or not file_urls:
            raise HttpApiError("MinerU 返回结构无效：缺少 batch_id / file_urls")
        trace["batch_id"] = batch_id
        trace["events"].append({"step": "create_batch", "batch_id": batch_id})
        return str(batch_id), str(file_urls[0])

    def _upload_file(
        self, upload_url: str, source_path: Path, trace: dict[str, Any]
    ) -> None:
        """第 2 步：PUT 上传原始字节。

        ★ 绝对不携带任何请求头（尤其 Content-Type）。预签名 URL 是按
        "空 Content-Type" 参与签名计算的，多带一个头就会被 OSS 判为
        `SignatureDoesNotMatch` 而 403。同理也不能带 Authorization。
        批次 22 实测：curl / http.client 不带 Content-Type → 200；
        urllib（自动补 Content-Type）或显式 Content-Type → 403。
        """
        content = source_path.read_bytes()
        self._call(
            "MinerU 上传文件",
            lambda: self.upload(upload_url, {}, content, self.timeout),
        )
        trace["events"].append({"step": "upload", "bytes": len(content)})

    def _await_result(self, batch_id: str, trace: dict[str, Any]) -> dict:
        """第 3 步：轮询任务结果，直到 done/failed 或超时。

        为什么用"总时长上限 + 固定间隔"而不是指数退避：
        MinerU 的解析耗时与页数成正比且不可预测，固定间隔能让总预算可控；
        退化保护靠 `max_poll_seconds`（超时抛 TimeoutError → 上层收敛成
        complete=False，再由 parse_pdf 决定是否走 Qwen-VL 兜底）。
        """
        url = f"{self.api_base_url}/api/v4/extract-results/batch/{batch_id}"
        headers = self._auth_headers()
        started = self.monotonic()
        polls = 0
        while True:
            response = self._call(
                "MinerU 轮询结果",
                lambda: self.transport_json(url, headers, b"", self.timeout),
            )
            result = self._extract_result_entry(response)
            state = result.get("state")
            polls += 1
            trace["events"].append(
                {
                    "step": "poll",
                    "n": polls,
                    "state": state,
                    "elapsed_seconds": round(self.monotonic() - started, 1),
                }
            )
            if state in _TERMINAL_STATES:
                trace["polls"] = polls
                trace["final_state"] = state
                return result
            if self.monotonic() - started >= self.max_poll_seconds:
                raise TimeoutError(
                    f"轮询超时（{self.max_poll_seconds}s，最后状态 {state}）"
                )
            self.sleep(self.poll_interval_seconds)

    def _extract_result_entry(self, response: dict) -> dict:
        """从轮询返回里取出 extract_result 的第一项，并校验结构。

        接口正常时 data.extract_result 是长度为 1 的数组（一次只提交一个文件）；
        结构不符时抛 HttpApiError，让 parse() 收敛成 complete=False 而不是
        抛出 KeyError/TypeError 这种"不像业务错误"的异常。
        """
        data = response.get("data") if isinstance(response, dict) else None
        results = data.get("extract_result") if isinstance(data, dict) else None
        if not isinstance(results, list) or not results:
            raise HttpApiError("MinerU 轮询返回结构无效：缺少 extract_result")
        entry = results[0]
        if not isinstance(entry, dict):
            raise HttpApiError("MinerU 轮询返回结构无效：extract_result 项非对象")
        return entry

    def _download_zip(self, zip_url: str, trace: dict[str, Any]) -> zipfile.ZipFile:
        """第 4 步：下载结果压缩包。

        下载地址是对象存储的预签名 URL，同样不带 Authorization。
        """
        def _fetch() -> bytes:
            return self.download(zip_url, {}, b"", self.timeout)

        archive_bytes = self._call("MinerU 下载结果", _fetch)
        trace["events"].append({"step": "download_zip", "bytes": len(archive_bytes)})
        return zipfile.ZipFile(io.BytesIO(archive_bytes))
