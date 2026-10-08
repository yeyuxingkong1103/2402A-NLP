"""FastAPI 应用装配。

**装配顺序是被锁定的，不是随意的：**

    中间件 → 异常处理器 → API 路由 → 静态挂载

⚠️ API 路由 MUST 在静态挂载**之前**注册。

Starlette 按**注册顺序**匹配路由。若先把 `frontend/` 挂到 `/`，那么 `/ask`
会被静态处理器接走，它的处理方式是"在 frontend/ 下找一个叫 ask 的文件"，
找不到就返回 404 —— 于是接口看起来像是"路由没写对"。

**而这个过程在启动时不报任何错。** 没有警告、没有异常，只有一个莫名其妙的
404。这是本模块存在的主要理由：把顺序固定下来并写明原因，避免后来者
"整理一下代码顺序"时把它拆掉。
"""

import logging
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from . import STATIC_DIR_NAME
from .config import PROJECT_ROOT, AppConfig, load
from .errors import register_exception_handlers
from .routes import router

logger = logging.getLogger(__name__)

__all__ = ["create_app", "NoCacheStaticFiles"]


class NoCacheStaticFiles(StaticFiles):
    """静态资源一律带 `Cache-Control: no-cache`。

    ⚠️ **这不是"开发期专用"的临时措施。**

    `no-cache` 不是"不缓存"—— 它是"用缓存之前先向服务端确认有没有变"。
    文件没变时服务端回 304，不重复传输；只有真的变了才重新下载。
    所以它的代价（一次协商往返）在有网络连接的场景下可以忽略，
    而它挡掉的问题很具体：

    **改了前端，用户刷新后看到的还是旧页面。**

    Starlette 的 `StaticFiles` 默认不发任何缓存头，浏览器于是用启发式规则
    长期缓存 JS/CSS。这在传统部署里靠文件名加 hash 解决（`app.a1b2c3.js`），
    而本项目的资源名是固定的 `app.js` / `nav.js` —— 没有那个机制可用。

    真实代价示例：删掉一个导航项后，源码里已经没有它了，浏览器侧栏却照旧
    显示，且**普通刷新无效** —— 排查方向会被引向"是不是没保存""是不是改错
    文件了"，而真相只是缓存。`routes.py` 给 SSE 响应加 `no-cache` 是同一条
    取向（那边防的是"看到上一个问题的答案"）。
    """

    def file_response(self, *args, **kwargs):  # type: ignore[override]
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache"
        return response


def create_app(config: AppConfig | None = None) -> FastAPI:
    """构造应用。`config` 缺省时自行加载（供 `serve.py` 之外的调用方使用）。"""

    config = config or load()
    static_dir = PROJECT_ROOT / STATIC_DIR_NAME

    app = FastAPI(
        title="医知源",
        # 关闭自动文档。
        #
        # 理由不是"少暴露信息"，而是**它会生成一份错误的契约**：
        # OpenAPI 会把这个接口描述成返回 JSON 的普通端点，而它实际返回
        # text/event-stream。一份自动生成的、与真实行为不符的文档，比没有
        # 文档更危险 —— 前端会照着它写解析代码。
        # 真实契约以 specs/006-medical-qa-input/contracts/sse.md 为准。
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    # 配置在启动期注入，请求路径上不再读环境变量（FR-022）。
    app.state.config = config

    # 1) 异常处理器 —— 必须早于路由，否则特殊异常路径会走框架默认输出。
    register_exception_handlers(app)

    # 2) API 路由 —— 必须在静态挂载之前，理由见模块文档字符串。
    app.include_router(router)

    # 3) 静态资源 —— 挂在根路径，`html=True` 让 `/` 返回 index.html。
    #
    # 用 NoCacheStaticFiles 而不是 StaticFiles，理由见那个类的文档字符串
    # （一句话：资源名没有内容指纹，默认缓存会让"改了前端但看不到变化"
    #   反复发生，而排查方向会被引向错误的地方）。
    if static_dir.is_dir():
        app.mount(
            "/",
            NoCacheStaticFiles(directory=str(static_dir), html=True),
            name="static",
        )
    else:
        # 不静默跳过：目录缺失会让页面 404，而原因（"没建 frontend/ 目录"）
        # 与现象（"接口通了但页面打不开"）之间没有可见的因果链。
        logger.warning(
            "静态资源目录不存在，界面将无法访问：%s（期望位置：%s）",
            static_dir,
            Path(STATIC_DIR_NAME),
        )

    return app
