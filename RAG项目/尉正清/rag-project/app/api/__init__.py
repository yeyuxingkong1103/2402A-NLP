# app/api/__init__.py
"""接口层。每个模块暴露 ROUTERS 列表，由 main.py 统一挂载。"""
from app.api import chat, knowledge, role, system

ROUTERS = chat.ROUTERS + role.ROUTERS + knowledge.ROUTERS + system.ROUTERS

__all__ = ["ROUTERS"]
