# -*- coding: utf-8 -*-
"""`legal_rag.api` 的模块级常量 —— 从 ``api/app.py`` 拆出。

为什么单独一个文件
------------------
这些常量原先散落在 ``api/app.py``（1,738 行）的顶部与中部（``_LLM_REASON_TEXT``
甚至在文件末尾）。路由模块与服务层都要用它们，集中一处才不必互相 import 造成环。

⚠️ 其中 :data:`PAGE_CACHE_CONTROL` 与 :data:`REQUEST_ID_HEADER` 是**行为契约**，
不是可调参数 —— 改动前请先读各自的注释与 ``tests/test_api.py`` 的对应用例。
"""
from __future__ import annotations

from pathlib import Path

__all__ = [
    "WEB_DIR", "UI_PATH", "AUTH_PAGE_FILES", "PAGE_CACHE_CONTROL",
    "REQUEST_ID_HEADER", "SNIFF_LIMIT", "REJECT_COUNTER", "LLM_REASON_TEXT",
]

#: 单页前端（无构建、零依赖，由本服务直接托管在 /ui）
WEB_DIR = Path(__file__).resolve().parents[2] / "web"
UI_PATH = WEB_DIR / "ui.html"

#: 三个认证页对应的前端文件（由前端任务创建；**缺失时回最小兜底页**，不是 404）
AUTH_PAGE_FILES = {"login": "login.html", "register": "register.html", "reset": "reset.html"}

#: 页面响应的缓存策略：**必须 `no-store`** —— 这不是可有可无的优化，是行为正确性的一部分。
#:
#: 为什么必须：`/ui` 与三个认证页都是**应用壳**（app shell），由后端 `FileResponse` 直接给出，
#: 而 `FileResponse` 默认**不带任何 `Cache-Control`**（只有 `ETag`/`Last-Modified`），浏览器于是可能
#: 在不重新验证的情况下复用旧 body。真机踩过（t19）：登出后仍看到旧的 `ui.html`，
#: 必须手加 `?t=` 查询串才刷新 —— 而**旧壳里没有跳转兜底与请求超时**，
#: 用户看到的就是"进对话页一直转圈、一直等待"。壳的新鲜度直接决定行为正确性
#: （新壳修好的 bug 会被旧壳复现），所以页面一律禁止"无重新验证的复用"。
#:
#: 取舍：**只加 `no-store`**，不叠 `no-cache, must-revalidate` —— `no-store` 已是最严格的
#: "不许存、不许复用"；对一个本就不该被存的响应再叠 `must-revalidate` 没有额外约束，
#: 只会让"到底哪条生效"变模糊（排障时反而更难定位）。
#: ⚠️ 业务 API **不**继承这个头（`tests/test_api.py::test_pages_forbid_reuse_but_api_does_not`
#: 是双向用例，防止以后被顺手扩大）：JSON 接口没有"壳"的语义，
#: 保持 FastAPI 默认的中立缓存语义，避免改坏 API 的缓存行为。
PAGE_CACHE_CONTROL = "no-store"

#: 每个响应都带这个头，便于把用户反馈的一条错误日志回溯到完整调用链
REQUEST_ID_HEADER = "X-Request-Id"

#: 单文件读入上限（比 max_upload_mb 稍大，超出的交由 validate_upload 分类拒绝）
SNIFF_LIMIT = 64 * 1024 * 1024

#: 上传被拒计数器：(指标名, 说明, 单位) —— 供 ``M.counter(*REJECT_COUNTER)`` 展开
REJECT_COUNTER = ("upload_rejected_total", "被拒的上传数（reason 分类）", "count")

#: LLM 失败原因 → 给用户看的中文（**不出现 host:port / 集合名 / 类名**，
#: 见 handoff/COPY-STANDARD.md）。取值来自
#: :func:`legal_rag.generate.llm_base.failure_reason`（机器口径），
#: 这里只负责翻译成人话；未命中的原因回落到 ``unknown`` 的通用文案。
#: ⚠️ 键名与文案是**逐字迁移**的（不是重写），改动等于改用户可见文案。
LLM_REASON_TEXT = {
    "missing_api_key": "大模型服务没有配置访问凭据",
    "connect_failed": "连不上大模型服务",
    "http_5xx": "大模型服务内部出错",
    "http_4xx": "大模型服务拒绝了这次请求",
    "timeout": "大模型服务响应超时",
    "model_missing": "大模型服务上没有这个模型",
    "auth": "大模型服务鉴权失败",
    "rate_limit": "大模型服务当前限流",
    "bad_request": "请求被大模型服务拒绝",
    "bad_response": "大模型服务返回的内容读不出来",
    "unknown": "大模型服务调用失败",
}
