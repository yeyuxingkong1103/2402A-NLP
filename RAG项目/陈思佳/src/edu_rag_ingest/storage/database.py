from __future__ import annotations

"""数据库 URL 和 SQLite 初始化辅助函数。"""

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatabaseConfig:
    url: str
    echo: bool = False


def default_database_url() -> str:
    return os.getenv("DATABASE_URL", "sqlite:///data/education_rag.db")


def ensure_sqlite_parent(url: str) -> None:
    prefix = "sqlite:///"
    if url.startswith(prefix) and url != "sqlite:///:memory:":
        Path(url[len(prefix) :]).parent.mkdir(parents=True, exist_ok=True)
