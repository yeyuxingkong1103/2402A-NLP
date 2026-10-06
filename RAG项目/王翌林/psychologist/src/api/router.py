"""v1 路由聚合。

这里是所有 v1 子路由的"总装配线"：每个子模块（auth、users 等）各自维护一个 APIRouter，
再由这个 api_router 统一 include 进来。这样做的好处是：
每个业务领域的路由文件可以独立维护、独立加注释，最终在这里一次性挂载到 /api/v1 前缀下。
"""
from fastapi import APIRouter

from src.api.v1 import admin, auth, chat, conversations, evaluate, knowledge, personas, users

# 统一前缀：所有子路由的路径最终都会拼接到 /api/v1 之后，
# 例如 auth 模块里的 /login 实际对外暴露为 /api/v1/auth/login。
api_router = APIRouter(prefix="/api/v1")
# include_router 只负责"挂载"，不改变子路由内部的路径、方法或依赖。
# 新增一个功能模块时，只要 import 它的 router 并在这里加一行即可。
api_router.include_router(auth.router)
api_router.include_router(users.router)
api_router.include_router(personas.router)
api_router.include_router(conversations.router)
api_router.include_router(chat.router)
api_router.include_router(knowledge.router)
api_router.include_router(admin.router)
api_router.include_router(evaluate.router)