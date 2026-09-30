"""MinerU 客户端 API 层：鉴权头、重试包装与三种默认传输实现。

从 `app.models.mineru` 按职责拆出（批次 25-2）。这一层只回答"请求怎么发出去"：
签名怎么带、失败怎么退避重试、预签名地址为什么不能带任何头；
"发什么内容、分几步发"属于业务流程，见 `app/models/mineru_flow.py`。

组合方式：`class MineruClient(MineruFlowMixin, MineruApiMixin)`。
`__init__` 仍留在 `app/models/mineru.py` —— 客户端状态只在一处初始化，
拆分不改变"谁能改哪个属性"。

★ 兼容性提醒：`tests/test_mineru_client.py` 里有
`MineruClient._request_upload.__get__(client)` 这种直接取方法的写法，
所以本模块的方法名与签名必须保持原样（方法体逐字未改）。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from app.models.http_retry import (
    HttpApiError,
    call_with_retry,
    request_json,
    request_raw,
)


class MineruApiMixin:
    """MinerU 的传输/重试/鉴权方法集合（供 `MineruClient` 组合）。

    本类不定义 `__init__`，依赖宿主类提供这些属性：
    `api_key` / `timeout` / `retry_attempts` / `retry_backoff_seconds` /
    `retry_total_budget_seconds` / `sleep` / `monotonic`。
    """

    # 由宿主类（MineruClient.__init__）注入的属性，这里只作**纯注解**声明
    # （不赋默认值 —— 赋了会让"漏初始化"从 AttributeError 变成静默用默认值）
    api_key: str
    timeout: float
    retry_attempts: int
    retry_backoff_seconds: float
    retry_total_budget_seconds: float
    sleep: Callable[[float], None]
    monotonic: Callable[[], float]

    def _auth_headers(self) -> dict[str, str]:
        """带 Bearer 的请求头；**只用于 MinerU 自己的接口**。

        预签名地址（上传 / 下载）绝不能带它：那些 URL 由对象存储校验签名，
        多带 Authorization 会被判为签名不符而 403。
        """
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "*/*",
        }

    def _call(self, what: str, func: Callable[[], Any]) -> Any:
        """统一走共用重试：超时/429/5xx 退避重试，400/401/403 立即抛。

        不可重试错误会被原样抛回（保留 status_code 供上层分类），
        这里只补一层「哪一步失败」的上下文 —— MinerU 有四个请求形态，
        错误信息里不写清是哪一步，排查时只能猜。
        """
        try:
            return call_with_retry(
                func,
                attempts=self.retry_attempts,
                backoff_seconds=self.retry_backoff_seconds,
                total_budget_seconds=self.retry_total_budget_seconds,
                what=what,
                sleep=self.sleep,
                monotonic=self.monotonic,
            )
        except HttpApiError as error:
            if str(error).startswith(what):
                raise  # 重试耗尽时 call_with_retry 已带 what 前缀，不重复拼
            raise HttpApiError(
                f"{what}失败：{error}", status_code=error.status_code
            ) from error

    # --------------------------------------------------------- 默认传输实现
    # 三个传输点都可被构造参数替换（单测注入假传输即可完全离线跑）。

    def _request_json(
        self, url: str, headers: dict[str, str], payload: bytes, timeout: float
    ) -> dict:
        """MinerU API 调用：有 body 是 POST，空 body 是 GET（轮询）。"""
        method = "POST" if payload else "GET"
        return request_json(url, headers, payload, timeout, method=method)

    def _request_upload(
        self, url: str, headers: dict[str, str], payload: bytes, timeout: float
    ) -> bytes:
        # 预签名地址：走 request_raw（只发给定的头，绝不自动补 Content-Type）
        return request_raw(
            url, method="PUT", payload=payload, headers=headers, timeout=timeout
        )

    def _request_download(
        self, url: str, headers: dict[str, str], payload: bytes, timeout: float
    ) -> bytes:
        # 结果包下载同样是预签名地址，规则同上传
        return request_raw(url, method="GET", headers=headers, timeout=timeout)
