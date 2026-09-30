# -*- coding: utf-8 -*-
"""Qdrant 嵌入式客户端单例。

⚠️ 嵌入式模式（QdrantClient(path=...)）对存储目录持有**独占文件锁**：
   同一时刻只允许一个进程打开。因此
   1. 后端必须以**单 worker** 启动（uvicorn 不要加 --workers）；
   2. 入库请走后端内的 /api/kb/ingest 接口，不要在服务运行时另起脚本再开一个客户端。
"""
import threading

from qdrant_client import QdrantClient

from ..core import config

_lock = threading.Lock()
_client: QdrantClient | None = None


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        with _lock:
            if _client is None:
                config.QDRANT_DIR.mkdir(parents=True, exist_ok=True)
                _client = QdrantClient(path=str(config.QDRANT_DIR))
    return _client


def close_client() -> None:
    global _client
    if _client is not None:
        try:
            _client.close()
        except Exception:
            pass
        _client = None
