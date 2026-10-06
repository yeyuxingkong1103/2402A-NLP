"""配置加载模块"""
import os
import yaml
from pathlib import Path
from typing import Any, Optional


class Config:
    def __init__(self, config_path: str):
        self._path = config_path
        self._data = {}
        self.load()

    def load(self):
        path = Path(self._path)
        if not path.exists():
            raise FileNotFoundError(f"配置文件不存在: {path}")
        with open(path, 'r', encoding='utf-8') as f:
            self._data = yaml.safe_load(f) or {}

    def get(self, key: str, default: Any = None) -> Any:
        val = self._data
        for k in key.split('.'):
            if isinstance(val, dict) and k in val:
                val = val[k]
            else:
                return default
        return val

    def __getitem__(self, key):
        return self.get(key)

    # ---- section getters（short_term.py 等需要） ----
    def get_redis_config(self) -> dict:
        return self._data.get('redis', {})

    def get_milvus_config(self) -> dict:
        return self._data.get('milvus', {})

    def get_llm_config(self) -> dict:
        return self._data.get('llm', {})

    def get_mysql_config(self) -> dict:
        return self._data.get('mysql', {})


_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_CONFIG_PATH = os.getenv("RAG_CONFIG", str(_PROJECT_ROOT / "config" / "config.yaml"))

config = Config(_CONFIG_PATH)

from src.utils.logger import logger, setup_logger

__all__ = ['config', 'logger', 'setup_logger']