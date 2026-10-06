"""FastAPI 应用入口。"""
from __future__ import annotations

# 这里是为了让「直接跑本文件」和「按模块跑」都能用：
#   python -m app.main   推荐，也是 PyCharm「Module name = app.main」运行配置的方式
#   python app/main.py   PyCharm 在本文件旁点绿三角时的默认方式，也是调试入口
# 后者会让本模块丢掉包上下文，下面的相对导入会报
# "attempted relative import with no known parent package"，所以先把包上下文补回来。
if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "app"

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .api import chat, health, knowledge, role
from .core.config import settings
from .core.logging_config import get_logger, setup_logging
from .core.pipeline import RAGPipeline
from .core.prompt.role_presets import ROLE_PRESETS

log = get_logger("main")

STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用启动/关闭钩子：启动时一次性组装整条 RAG 链路。"""
    setup_logging(settings)
    log.info(
        "配置层: %s%s",
        " ← ".join(settings.env_files) or "(未读到任何 .env 文件，全部走内置默认值)",
        f"（APP_ENV={settings.app_env}）" if settings.app_env else "",
    )
    # 组装 LLM / embedding / 重排 / 向量库 / 关系库 / 记忆，之后每次请求都复用这个 pipeline
    app.state.pipeline = RAGPipeline.build(settings)
    # 把内置预设角色登记/补齐进库（含头像），保证 /role/list 一上来就是完整的。
    #
    # 这一步要写关系库，是启动期**唯一**的急切依赖：MySQL 未起/网络不通时这里会抛
    # OperationalError，uvicorn 直接 "Application startup failed" 退出——nginx 后面
    # 所有 worker 一起死，而 /health 里那条为 sql 专门留的 unavailable/degraded 档
    # 永远显示不出来。其它组件（LLM/向量库/记忆）都是懒连接 + 降级，这里跟着对齐：
    # 起不来就记一条 error（错误信息里带原因），服务照常提供 /health 与其它接口。
    for preset_id in ROLE_PRESETS:
        try:
            app.state.pipeline.ensure_role(preset_id)
        except Exception as exc:  # noqa: BLE001
            log.error(
                "注册预设角色失败（关系库不可用？服务继续启动，/health 会报 sql 不可达）: %s", exc
            )
            break
    log.info("RAG 角色扮演系统启动完成")
    yield


app = FastAPI(
    title="RAG 角色扮演系统",
    description="基于 RAG 的多角色对话系统（MVP）",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(chat.router)
app.include_router(knowledge.router)
app.include_router(role.router)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def root(request: Request):
    """内容协商：浏览器拿到聊天页，脚本 / 客户端仍拿到原来的 JSON。"""
    if "text/html" in request.headers.get("accept", ""):
        return FileResponse(STATIC_DIR / "index.html")
    return {"service": "rag-roleplay", "version": "0.1.0", "docs": "/docs", "web": "/"}


if __name__ == "__main__":
    # 用 `python -m app.main` 启动（在项目根目录下执行）。
    # 不要用 `python app/main.py`：那会让本模块的相对导入失效。
    import os

    import uvicorn

    uvicorn.run(
        app,
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8000")),
    )
