"""
模型单例管理器
避免每次请求都重新初始化 Embedding / Rerank / Chat 模型。
"""
import threading
import logging
from typing import Dict, Any, Callable

_lock = threading.Lock()
_model_cache: Dict[str, Any] = {}


def get_or_create(key: str, factory: Callable[[], Any]) -> Any:
    """线程安全的懒加载单例"""
    if key in _model_cache:
        return _model_cache[key]

    with _lock:
        if key in _model_cache:
            return _model_cache[key]
        logging.info(f"[singleton] creating model: {key}")
        _model_cache[key] = factory()
        return _model_cache[key]


def clear(key: str = None):
    """清空缓存"""
    global _model_cache
    if key:
        _model_cache.pop(key, None)
    else:
        _model_cache.clear()


def stats() -> Dict[str, str]:
    """返回当前缓存的模型清单"""
    return {k: type(v).__name__ for k, v in _model_cache.items()}