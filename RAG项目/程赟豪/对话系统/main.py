"""主入口"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from src.config import config
from src.utils.logger import logger, setup_logger
from src.api.routes import router


def create_app() -> FastAPI:
    app = FastAPI(
        title=config.get("app.name", "RAG角色扮演系统"),
        version=config.get("app.version", "1.0.0"),
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)

    # 前端静态资源与页面
    static_dir = PROJECT_ROOT / "static"
    static_dir.mkdir(exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/", include_in_schema=False)
    async def index():
        """前端首页"""
        return FileResponse(str(static_dir / "index.html"))

    return app


app = create_app()


if __name__ == "__main__":
    # 初始化日志
    try:
        setup_logger()
    except Exception:
        pass

    host = config.get("app.host", "0.0.0.0")
    port = config.get("app.port", 8000)
    debug = config.get("app.debug", False)

    logger.info(f"启动服务: http://{host}:{port}")
    uvicorn.run("main:app", host=host, port=port, reload=debug)