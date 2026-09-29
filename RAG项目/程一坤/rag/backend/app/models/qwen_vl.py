"""Qwen-VL 适配器：PDF 渲染成页图后做文字识别，作为 MinerU 的兜底。

使用场景：MinerU 返回不完整（任务失败 / 超时 / 正文为空）时，
`app/ingest/pdf_parser.parse_pdf` 会用本客户端再试一次 —— 它对扫描件、
MinerU 不支持的版面也能出文字，代价是没有结构、只有纯文本。

调用契约（与 parse_pdf 对齐）：
    QwenVlClient.parse(source_path) -> str   # 拼接后的页面文本
失败时返回空串并把原因写进 `last_error`（parse_pdf 会据此判「都没有 → PdfParsingError」）。

页面渲染抽成可注入的 `renderer`：默认实现用 PyMuPDF（fitz）把每页渲染成 PNG；
单测传入假 renderer 即可在**不安装 PyMuPDF** 的环境里覆盖全部逻辑。
"""

from __future__ import annotations

import base64
import json
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from app.models.http_retry import HttpApiError, Transport, call_with_retry, chat_completions_endpoint, request_json

logger = logging.getLogger(__name__)

# 页渲染函数签名：PDF 路径 → 每页 PNG 字节
PageRenderer = Callable[[Path], list[bytes]]

# qwen-vl-ocr 官方说明：该模型内部固定使用 "Read all the text in the image."，
# 用户传入的文本不生效。这里仍带上文本项，是为了保持请求结构合法。
_OCR_INSTRUCTION = "Read all the text in the image."

# 图片缩放范围（DashScope 文档推荐值）：太小会丢字，太大浪费额度
_MIN_PIXELS = 3072
_MAX_PIXELS = 8388608

# 渲染分辨率：150 DPI 在"正文可辨认"与"请求体积可控"之间取平衡
_DEFAULT_DPI = 150


class QwenVlClient:
    """DashScope 兼容模式下 qwen-vl-ocr 的客户端。"""

    def __init__(
        self,
        api_base_url: str,
        api_key: str,
        model: str = "qwen-vl-ocr",
        *,
        timeout: float = 120.0,
        max_pages_per_request: int = 8,
        max_request_bytes: int = 1_500_000,
        dpi: int = _DEFAULT_DPI,
        renderer: PageRenderer | None = None,
        transport: Transport | None = None,
        retry_attempts: int = 3,
        retry_backoff_seconds: float = 0.5,
        retry_total_budget_seconds: float = 20.0,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.api_base_url = api_base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_pages_per_request = max(1, int(max_pages_per_request))
        # 单请求体积上限（base64 后的字符数）。★ 这不是省流量，是正确性问题：
        # 实测 qwen-vl-ocr 的请求体超过约 2MB 时**只返回第一张图的内容**：
        #   4 图（1.66 MB）→ 1929 字（完整）
        #   6 图（2.45 MB）→  400 字（只剩第 1 页）
        #   8 图（3.28 MB）→  399 字（只剩第 1 页）
        #   逐页发送 8 张 → 3816 字（完整，质量最好）
        # 所以批次由「体积」和「页数」双重约束，体积是真正生效的那个。
        self.max_request_bytes = max(1, int(max_request_bytes))
        self.dpi = dpi
        # 渲染与传输都可替换：单测不依赖 PyMuPDF，也不联网
        self.renderer = renderer or self._render_pages
        self.transport = transport or self._request
        self.retry_attempts = retry_attempts
        self.retry_backoff_seconds = retry_backoff_seconds
        self.retry_total_budget_seconds = retry_total_budget_seconds
        self.sleep = sleep
        self.monotonic = monotonic
        # 最近一次失败原因（parse 返回空串时来这里取原因）
        self.last_error = ""
        # 最近一次 parse 的执行轨迹（页数 / 批次数 / 每批字符数 / 耗时），仅用于诊断
        self.last_trace: dict[str, Any] = {}

    @property
    def chat_endpoint(self) -> str:
        """对话接口地址；容忍配置里带或不带结尾斜杠。"""
        return chat_completions_endpoint(self.api_base_url)

    # ------------------------------------------------------------------ 对外

    def parse(self, source_path: Path) -> str:
        """渲染每一页并做 OCR，返回拼接后的文本；失败返回空串。"""
        started = self.monotonic()
        trace: dict[str, Any] = {
            "file": source_path.name,
            "model": self.model,
            "batches": [],
        }
        self.last_trace = trace
        self.last_error = ""

        if not source_path.is_file():
            return self._failed(f"PDF 文件不存在：{source_path}", trace, started)

        try:
            images = self.renderer(source_path)
        except ImportError as error:
            return self._failed(f"缺少页面渲染依赖：{error}", trace, started)
        except (OSError, ValueError, RuntimeError) as error:
            return self._failed(
                f"PDF 页面渲染失败（{type(error).__name__}）：{error}", trace, started
            )

        if not images:
            return self._failed("PDF 未渲染出任何页面", trace, started)

        trace["page_count"] = len(images)
        chunks: list[str] = []
        for start, end in self._plan_batches(images, trace):
            batch = images[start:end]
            try:
                text = self._ocr_batch(batch, trace)
            except HttpApiError as error:
                # 重试耗尽后的超时/429/5xx 都会收敛成 HttpApiError（见 http_retry）
                return self._failed(f"Qwen-VL 调用失败：{error}", trace, started)
            except (KeyError, TypeError, ValueError) as error:
                return self._failed(
                    f"Qwen-VL 返回结构无效：{type(error).__name__}: {error}", trace, started
                )
            if text.strip():
                chunks.append(text.strip())

        content = "\n".join(chunks)
        trace["content_chars"] = len(content)
        trace["elapsed_seconds"] = round(self.monotonic() - started, 2)
        if not content.strip():
            return self._failed("Qwen-VL 未识别出任何文字", trace, started)
        return content

    # ------------------------------------------------------------- 内部步骤

    def _plan_batches(
        self, images: list[bytes], trace: dict[str, Any]
    ) -> list[tuple[int, int]]:
        """把页图切成若干批次，返回 [(起, 止)]。

        双重约束（任一先触发就断批）：
        1. 体积：累计 base64 字符数不超过 max_request_bytes —— **真正生效的那个**，
           超了会让服务端只回第一页内容（详见 __init__ 里的实测数据）；
        2. 页数：不超过 max_pages_per_request（配置上限，防止意外提交超大请求）。

        单页体积本身就超上限时不拆（一页拆不了），单独成批并记警告。
        """
        overhead = len("data:image/png;base64,")
        # base64 长度公式：每 3 字节 → 4 字符，不足补 4 的倍数
        sizes = [overhead + ((len(image) + 2) // 3) * 4 for image in images]

        batches: list[tuple[int, int]] = []
        start = 0
        current_bytes = 0
        for index, size in enumerate(sizes):
            count = index - start
            exceeds_bytes = current_bytes + size > self.max_request_bytes
            exceeds_pages = count >= self.max_pages_per_request
            if count > 0 and (exceeds_bytes or exceeds_pages):
                batches.append((start, index))
                start = index
                current_bytes = 0
            current_bytes += size

        if sizes and sizes[start] > self.max_request_bytes:
            logger.warning(
                "单页图 base64 体积 %d 超过单请求上限 %d，该页单独发送；"
                "若仍失败请调低 QWEN_VL_RENDER_DPI",
                sizes[start],
                self.max_request_bytes,
            )
        if start < len(sizes):
            batches.append((start, len(sizes)))

        trace["batches_planned"] = [
            {"images": end - begin, "payload_bytes": sum(sizes[begin:end])}
            for begin, end in batches
        ]
        return batches

    def _ocr_batch(self, images: list[bytes], trace: dict[str, Any]) -> str:
        """把一批页图发给 qwen-vl-ocr，返回该批识别文本。"""
        content: list[dict[str, Any]] = []
        for image in images:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64,"
                        + base64.b64encode(image).decode("ascii")
                    },
                    "min_pixels": _MIN_PIXELS,
                    "max_pixels": _MAX_PIXELS,
                }
            )
        content.append({"type": "text", "text": _OCR_INSTRUCTION})

        payload = json.dumps(
            {
                "model": self.model,
                "messages": [{"role": "user", "content": content}],
            },
            ensure_ascii=False,
        ).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        response = call_with_retry(
            lambda: self.transport(self.chat_endpoint, headers, payload, self.timeout),
            attempts=self.retry_attempts,
            backoff_seconds=self.retry_backoff_seconds,
            total_budget_seconds=self.retry_total_budget_seconds,
            what="Qwen-VL 文字识别",
            sleep=self.sleep,
            monotonic=self.monotonic,
        )
        text = self._extract_text(response)
        trace["batches"].append(
            {"images": len(images), "payload_bytes": len(payload), "chars": len(text)}
        )
        return text

    @staticmethod
    def _extract_text(response: dict) -> str:
        """从兼容模式返回结构中取出文本。"""
        # 故意用下标而不是 .get()：结构不符时抛 KeyError/TypeError，
        # 由调用方 _ocr_batch 统一收敛成"返回结构无效"（好过静默返回空串）
        choices = response["choices"]
        message = choices[0]["message"]
        text = message.get("content")
        # 部分多模态返回把正文放在数组里（[{"type":"text","text":"..."}]）
        if isinstance(text, list):
            text = "".join(
                part.get("text", "") for part in text if isinstance(part, dict)
            )
        if text is None:
            return ""
        return str(text)

    def _render_pages(self, source_path: Path) -> list[bytes]:
        """默认渲染实现：用 PyMuPDF 把每页渲染成 PNG 字节。"""
        try:
            import fitz  # PyMuPDF；延迟导入，非 PDF 链路不受影响
        except ImportError as error:  # pragma: no cover - 依赖缺失时的明确报错
            raise ImportError(
                "Qwen-VL 兜底需要 PyMuPDF 渲染页图，请安装 PyMuPDF"
            ) from error

        zoom = self.dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        pages: list[bytes] = []
        with fitz.open(str(source_path)) as document:
            for page in document:
                pixmap = page.get_pixmap(matrix=matrix, alpha=False)
                pages.append(pixmap.tobytes("png"))
        return pages

    def _failed(self, message: str, trace: dict[str, Any], started: float) -> str:
        """收敛失败：记轨迹 + 记原因 + 记日志，返回空串。"""
        trace["error"] = message
        trace["elapsed_seconds"] = round(self.monotonic() - started, 2)
        self.last_error = message
        logger.warning("Qwen-VL 兜底未完成：%s", message)
        return ""

    def _request(
        self, url: str, headers: dict[str, str], payload: bytes, timeout: float
    ) -> dict:
        return request_json(url, headers, payload, timeout, method="POST")


def build_qwen_vl_client_from_settings() -> QwenVlClient:
    """按应用配置创建 Qwen-VL 客户端。"""
    from app.core.config import settings

    return QwenVlClient(
        api_base_url=settings.qwen_vl_api_base_url,
        api_key=settings.qwen_vl_api_key,
        model=settings.qwen_vl_model,
        timeout=settings.qwen_vl_timeout_seconds,
        max_pages_per_request=settings.qwen_vl_max_pages_per_request,
        max_request_bytes=settings.qwen_vl_max_request_bytes,
    )
