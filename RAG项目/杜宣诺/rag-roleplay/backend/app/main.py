from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .api.routes import auth, characters, chat, documents, favorites, memories, sessions
from .db.base import Base, engine
from .db import models  # noqa: F401  确保模型已注册到 Base.metadata
from .deps import get_milvus


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 本地开发：启动时自动建表（生产可替换为 alembic 迁移）
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    # 确保 Milvus collection 存在（幂等）
    await get_milvus().init_collections()
    yield


def create_app() -> FastAPI:
    app = FastAPI(title=get_settings().app_name, lifespan=lifespan)

    # 自部署场景：允许浏览器跨域调用（前端可独立托管或用 file:// 打开）
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/v1/health")
    async def health():
        return {"code": 0, "data": {"status": "ok"}, "message": "ok"}

    app.include_router(auth.router)
    app.include_router(auth.users_router)
    app.include_router(characters.router)
    app.include_router(sessions.router)
    app.include_router(chat.router)
    app.include_router(memories.router)
    app.include_router(documents.router)
    app.include_router(favorites.router)
    return app


app = create_app()
