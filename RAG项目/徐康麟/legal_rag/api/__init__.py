# -*- coding: utf-8 -*-
"""FastAPI HTTP JSON 接口。"""
from .app import AppState, create_app  # noqa: F401

__all__ = ["create_app", "AppState"]
