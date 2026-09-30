"""MinerU 适配器：把本地 PDF 交给 MinerU 做结构化解析，取回 Markdown 正文。

调用契约（与 `app/ingest/pdf_parser.parse_pdf` 对齐）：
    MineruClient.parse(source_path) -> {
        "content": str,      # 解析出的正文（Markdown）
        "page_count": int,   # 页数
        "complete": bool,    # 是否拿到完整结果
        "error": str,        # 不完整时的原因（完整时为空串）
    }

关键约定：**本客户端不抛异常**。任何失败（网络、鉴权、超时、任务 failed、
zip 里没有正文）都收敛成 `complete=False` + `error=原因`，由 `parse_pdf`
决定是否退化到 Qwen-VL 兜底。理由：MinerU 挂掉不该让整条 PDF 通道失败。

MinerU v4 的四步流程：
1) POST {base}/api/v4/file-urls/batch        （带 Bearer）→ batch_id + 预签名上传地址
2) PUT  预签名地址  上传 PDF 字节            （★ 绝不带 Authorization，带了会 403）
3) GET  {base}/api/v4/extract-results/batch/{batch_id}  轮询 → state ∈ pending/running/done/failed
4) 下载 full_zip_url 并解压，取 full.md 作为正文

模块拆分说明（批次 25-2）：
本文件原 446 行，超过 `docs/目录与命名约定.md` §3.4 的单文件 300 行上限，
按职责拆成四块 —— 本文件保留「常量 + 类装配 + 对外入口」：

- `app/models/mineru_api.py`    ← 客户端 API 层：鉴权头 / 重试包装 / 三种默认传输
- `app/models/mineru_flow.py`   ← 上传与轮询下载流程：v4 第 1~4 步
- `app/models/mineru_result.py` ← 结果解析：正文与页数
- 本文件：`MAX_FILE_BYTES` / `MAX_PAGES` + `MineruClient.__init__` +
  `parse()`（编排入口）+ `_build_payload()` + `_failed()` +
  `build_mineru_client_from_settings()`

两个"必须留在本文件"的东西及原因：
1. `MAX_FILE_BYTES` 与读它的 `_build_payload()`：`tests/test_mineru_client.py`
   用 `monkeypatch.setattr("app.models.mineru.MAX_FILE_BYTES", 0)` 触发尺寸拦截，
   若常量与读它的函数分处两个模块，patch 只会改本模块的名字、读到的仍是旧值，
   测试会**静默失效**（不是报错，是"看起来还是通过"）。
2. `__init__`：客户端状态的唯一初始化点，拆开会让"谁定义了什么"变得难追。
"""

from __future__ import annotations

import json
import logging
import time
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from app.models.http_retry import BytesTransport, HttpApiError, Transport
from app.models.mineru_api import MineruApiMixin
from app.models.mineru_flow import MineruFlowMixin
from app.models.mineru_result import extract_markdown, resolve_page_count

logger = logging.getLogger(__name__)

# MinerU 官方限制：文件大小在 `_build_payload()` 处提前拦下，省一次无效调用
MAX_FILE_BYTES = 200 * 1024 * 1024
# ⚠️ 当前**未实现**页数拦截（MinerU 超 200 页会被服务端拒绝），本常量只登记官方上限；
# 补实现方案见 `reports/b41_deadcode_cleanup.md` 待办段（部署验收后处理）
MAX_PAGES = 200


class MineruClient(MineruFlowMixin, MineruApiMixin):
    """MinerU v4 API 客户端。

    具体能力由两个 mixin 提供（分层见模块 docstring）：
    - `MineruFlowMixin`：第 1~4 步流程方法
    - `MineruApiMixin` ：鉴权头 / 重试包装 / 三种传输实现

    本类自己负责三件事：初始化全部状态、对外入口 `parse()`、失败收敛 `_failed()`。
    """

    def __init__(
        self,
        api_base_url: str,
        api_key: str,
        *,
        poll_interval_seconds: float = 3.0,
        max_poll_seconds: float = 300.0,
        language: str = "ch",
        enable_table: bool = True,
        enable_formula: bool = True,
        is_ocr: bool = False,
        model_version: str = "vlm",
        timeout: float = 60.0,
        transport_json: Transport | None = None,
        upload: BytesTransport | None = None,
        download: BytesTransport | None = None,
        retry_attempts: int = 3,
        retry_backoff_seconds: float = 0.5,
        retry_total_budget_seconds: float = 15.0,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.api_base_url = api_base_url.rstrip("/")
        self.api_key = api_key
        self.poll_interval_seconds = poll_interval_seconds
        self.max_poll_seconds = max_poll_seconds
        self.language = language
        self.enable_table = enable_table
        self.enable_formula = enable_formula
        self.is_ocr = is_ocr
        self.model_version = model_version
        self.timeout = timeout
        # 三个传输点都可替换：POST(JSON) / PUT(上传) / GET(下载)
        self.transport_json = transport_json or self._request_json
        self.upload = upload or self._request_upload
        self.download = download or self._request_download
        self.retry_attempts = retry_attempts
        self.retry_backoff_seconds = retry_backoff_seconds
        self.retry_total_budget_seconds = retry_total_budget_seconds
        self.sleep = sleep
        self.monotonic = monotonic
        # 最近一次 parse 的执行轨迹（batch_id / 轮询状态流转 / 耗时），
        # 只用于诊断与验收取证，不参与业务逻辑
        self.last_trace: dict[str, Any] = {}

    # ------------------------------------------------------------------ 对外

    def parse(self, source_path: Path) -> dict:
        """解析 PDF；任何失败都返回 complete=False 而不是抛异常。

        参数：
            source_path: 本地 PDF 路径
        返回：
            `{"content": str, "page_count": int, "complete": bool, "error": str}`
            （契约见模块 docstring）；同时把执行轨迹写进 `self.last_trace`。

        为什么所有异常都在这里收敛成返回值：调用方 `parse_pdf` 需要在
        "MinerU 挂了"时继续走 Qwen-VL 兜底，用返回值表达比用异常更好判断，
        也避免把 `HttpApiError` / `TimeoutError` 这些实现细节泄漏给版面层。
        """
        started = self.monotonic()
        trace: dict[str, Any] = {"file": source_path.name, "events": []}
        self.last_trace = trace

        # 本地体检（文件存在 + 尺寸）放在最前：不合法就没必要占用一次配额
        try:
            payload = self._build_payload(source_path)
        except (OSError, ValueError) as error:
            return self._failed(f"{type(error).__name__}: {error}", trace, started)

        # 四步调用链：提交 → 上传 → 轮询（→ 失败态也在这里返回）
        try:
            batch_id, upload_url = self._create_batch(payload, trace)
            self._upload_file(upload_url, source_path, trace)
            result = self._await_result(batch_id, trace)
        except HttpApiError as error:
            return self._failed(f"MinerU API 错误：{error}", trace, started)
        except TimeoutError as error:
            return self._failed(f"MinerU 请求超时：{error}", trace, started)
        except (OSError, ValueError) as error:
            return self._failed(f"{type(error).__name__}: {error}", trace, started)

        # 任务跑完但状态是 failed：把服务端给的 err_msg 原样带上，便于排查
        if result.get("state") == "failed":
            return self._failed(
                f"MinerU 任务失败：{result.get('err_msg') or '未给出原因'}", trace, started
            )

        zip_url = result.get("full_zip_url")
        if not zip_url:
            return self._failed("MinerU 任务完成但未返回下载地址", trace, started)

        # 第 4 步：下载结果包并取正文（下载失败同样不抛，收敛成 complete=False）
        try:
            archive = self._download_zip(zip_url, trace)
            content = extract_markdown(archive)
        except (HttpApiError, TimeoutError, OSError, ValueError, zipfile.BadZipFile) as error:
            return self._failed(f"下载/解压 MinerU 结果失败：{error}", trace, started)

        page_count = resolve_page_count(result, archive, trace)

        # 正文全空白视为不完整：有 Markdown 结构但没字的包对下游没用
        if not content.strip():
            return self._failed("MinerU 返回的正文为空", trace, started)

        trace["complete"] = True
        trace["elapsed_seconds"] = round(self.monotonic() - started, 2)
        trace["content_chars"] = len(content)
        return {
            "content": content,
            "page_count": page_count,
            "complete": True,
            "error": "",
        }

    # ------------------------------------------------------------- 内部步骤

    def _build_payload(self, source_path: Path) -> bytes:
        """构造提交任务的请求体；本地先做尺寸/页数体检，避免无效调用。

        参数：
            source_path: 本地 PDF 路径
        返回：
            UTF-8 编码的 JSON 请求体
        异常：
            ValueError: 文件不存在或超过 `MAX_FILE_BYTES`（由 parse 收敛）
        """
        if not source_path.is_file():
            raise ValueError(f"PDF 文件不存在：{source_path}")
        size = source_path.stat().st_size
        if size > MAX_FILE_BYTES:
            raise ValueError(
                f"PDF 超过 MinerU 限制（{size} > {MAX_FILE_BYTES} 字节）"
            )

        body = {
            "files": [
                {
                    "name": source_path.name,
                    "is_ocr": self.is_ocr,
                    "data_id": source_path.stem,
                }
            ],
            "language": self.language,
            "enable_table": self.enable_table,
            "enable_formula": self.enable_formula,
            "model_version": self.model_version,
        }
        return json.dumps(body, ensure_ascii=False).encode("utf-8")

    def _failed(self, message: str, trace: dict[str, Any], started: float) -> dict:
        """收敛失败结果：写轨迹、记日志、返回 complete=False。"""
        trace["complete"] = False
        trace["error"] = message
        trace["elapsed_seconds"] = round(self.monotonic() - started, 2)
        logger.warning("MinerU 解析未完成：%s", message)
        return {"content": "", "page_count": 0, "complete": False, "error": message}


def build_mineru_client_from_settings() -> MineruClient:
    """按应用配置创建 MinerU 客户端。"""
    from app.core.config import settings

    return MineruClient(
        api_base_url=settings.mineru_api_base_url,
        api_key=settings.mineru_api_key,
        poll_interval_seconds=settings.mineru_poll_interval_seconds,
        max_poll_seconds=settings.mineru_max_poll_seconds,
        language=settings.mineru_language,
        enable_table=settings.mineru_enable_table,
        enable_formula=settings.mineru_enable_formula,
        is_ocr=settings.mineru_is_ocr,
    )
