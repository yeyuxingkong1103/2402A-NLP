# -*- coding: utf-8 -*-
"""统一日志系统。

所有模块通过 ``get_logger(name)`` 取到 ``rag2.<name>`` 命名空间的 logger；
启动时调用一次 ``setup_logging()`` 配置 root ``rag2`` logger：
控制台输出 + ``RotatingFileHandler`` 落盘到 ``logs/rag2.log``（默认）。

用法：
    from rag2 import setup_logging, get_logger

    setup_logging(level="INFO", log_dir="logs")
    logger = get_logger("server")
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

ROOT_LOGGER = "rag2"

_DEFAULT_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_configured = False


def get_logger(name: str | None = None) -> logging.Logger:
    """获取 ``rag2.<name>`` 命名空间的 logger。

    参数：
        name: 子模块名；``None`` / ``"rag2"`` / ``"__main__"`` 时返回 root ``rag2``。
    """
    if not name or name in ("rag2", "__main__", "root"):
        return logging.getLogger(ROOT_LOGGER)
    if name.startswith("rag2."):
        return logging.getLogger(name)
    return logging.getLogger(f"{ROOT_LOGGER}.{name}")


def setup_logging(
    level: str | int = "INFO",
    log_dir: str | Path = "logs",
    log_file: str | Path | None = None,
    console: bool = True,
    max_bytes: int = 5 * 1024 * 1024,
    backup_count: int = 5,
) -> logging.Logger:
    """配置 root ``rag2`` logger（幂等：重复调用不会重复添加 handler）。

    参数：
        level:        日志级别（"DEBUG"/"INFO"/... 或 int）。
        log_dir:      日志目录（相对路径以当前工作目录为基准）。
        log_file:     日志文件名；缺省为 ``<log_dir>/rag2.log``；传 ``"-"`` 表示不落盘。
        console:      是否同时输出到控制台。
        max_bytes:    单个日志文件上限，超出滚动。
        backup_count: 滚动保留的备份文件数。
    """
    global _configured
    root = logging.getLogger(ROOT_LOGGER)
    if _configured:
        root.setLevel(level if isinstance(level, int) else str(level).upper())
        return root
    _configured = True

    root.setLevel(level if isinstance(level, int) else str(level).upper())
    root.propagate = False
    formatter = logging.Formatter(_DEFAULT_FORMAT, datefmt=_DATE_FORMAT)

    if console:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(formatter)
        root.addHandler(handler)

    # RotatingFileHandler：日志文件超过 max_bytes 自动「滚动」——把旧日志改名备份（.1/.2…），
    # 最多保留 backup_count 份，避免日志无限增长占满磁盘。
    if log_file != "-":
        path = Path(log_file) if log_file else (Path(log_dir) / "rag2.log")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = RotatingFileHandler(
                str(path), maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
            )
            file_handler.setFormatter(formatter)
            root.addHandler(file_handler)
        except OSError as exc:  # 落盘失败不影响程序运行
            root.warning("日志文件创建失败，仅输出到控制台：%s", exc)

    return root
