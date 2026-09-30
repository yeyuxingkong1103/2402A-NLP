# -*- coding: utf-8 -*-
"""FastAPI 入口：注册路由、连接 Redis、建表并写入内置角色。"""
import logging
# 解析：日志模块
from contextlib import asynccontextmanager
# 解析：异步上下文管理器（生命周期钩子）
from pathlib import Path
# 解析：路径处理

import redis
# 解析：redis-py 客户端
from fastapi import FastAPI
# 解析：FastAPI 框架
from fastapi.responses import FileResponse
# 解析：文件响应（返回 index.html）
from fastapi.staticfiles import StaticFiles
# 解析：静态文件托管（前端 CSS/JS）

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
# 解析：前端目录绝对路径（项目根/frontend）

from app.api import chat_knowledge, users_roles
# 解析：路由模块（对话+知识库 / 用户+角色）
from app.config import settings
# 解析：全局配置
from app.models.db import SessionLocal, init_db
# 解析：数据库会话与建表函数
from app.seed import seed_roles, upgrade_default_prompt_templates
# 解析：种子数据与模板升级

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
# 解析：全局日志格式（时间/级别/模块/消息）
logger = logging.getLogger("rag-roleplay")
# 解析：根 logger


# 应用工厂：注册路由 + Redis 连接 + 建表/种子（init_prod_db 控制生产初始化）
def create_app(init_prod_db: bool = True) -> FastAPI:
    # 解析：应用工厂（测试传 False 跳过生产初始化）
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # 解析：应用生命周期（启动/关闭钩子）
        app.state.redis = redis.Redis(
            # 解析：创建全局 Redis 客户端
            host=settings.redis_host,
            # 解析：地址
            port=settings.redis_port,
            # 解析：端口
            password=settings.redis_password or None,
            # 解析：密码（空串转 None）
            db=settings.redis_db,
            # 解析：库号
            decode_responses=True,
            # 解析：响应自动解码为字符串
        )
        if init_prod_db:
            # 解析：生产初始化
            init_db()
            # 解析：建表（幂等）
            with SessionLocal() as db:
                # 解析：开启会话
                added = seed_roles(db)
                # 解析：写入内置角色
                if added:
                    # 解析：有新角色
                    logger.info("首次启动，已写入 %d 个内置角色", added)
                    # 解析：记录日志
                upgraded = upgrade_default_prompt_templates(db)
                # 解析：旧模板升级
                if upgraded:
                    # 解析：有升级
                    logger.info("已将 %d 个角色的旧默认模板升级（含知识库占位符）", upgraded)
                    # 解析：记录日志
        yield
        # 解析：应用运行中
        app.state.redis.close()
        # 解析：关闭 Redis 连接

    app = FastAPI(title="RAG 角色扮演系统", version="0.2.0", lifespan=lifespan)
    # 解析：创建 FastAPI 应用（含生命周期钩子）
    app.include_router(users_roles.user_router)
    # 解析：注册/登录路由
    app.include_router(users_roles.role_router)
    # 解析：角色 CRUD 路由
    app.include_router(chat_knowledge.chat_router)
    # 解析：对话路由
    app.include_router(chat_knowledge.kb_router)
    # 解析：知识库路由

    # 前端单页应用：/ 返回 index.html，静态资源挂 /static（须在 API 路由之后挂载）
    if FRONTEND_DIR.exists():
        # 解析：前端目录存在
        app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
        # 解析：静态资源挂载（CSS/JS）

        @app.get("/", include_in_schema=False)
        # 解析：根路径路由（不进 Swagger）
        def frontend_index():
            # 解析：返回前端页面
            return FileResponse(FRONTEND_DIR / "index.html")
            # 解析：返回 index.html

    return app
    # 解析：返回应用实例


app = create_app()
# 解析：模块级应用实例（uvicorn app.main:app 启动入口）
