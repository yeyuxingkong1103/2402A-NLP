# -*- coding: utf-8 -*-
"""日志模块：python logging，控制台 + 滚动文件双输出。"""
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from config import APP_ENV, LOG_DIR


def setup_logging(name: str = "rag_roleplay") -> logging.Logger:
    """初始化全局 logger：开发环境 DEBUG 级，其他 INFO 级；幂等可重复调用。"""
    logger = logging.getLogger(name)
    if logger.handlers:  # 已有 handler 说明初始化过，直接复用
        return logger
    logger.setLevel(logging.DEBUG if APP_ENV == "development" else logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    stream = logging.StreamHandler()  # 输出到控制台
    stream.setFormatter(fmt)
    file_handler = RotatingFileHandler(  # 输出到文件：单文件 2MB，最多保留 5 个备份
        Path(LOG_DIR) / "app.log",
        maxBytes=2_000_000,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(fmt)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


# 全局共享日志对象，各模块 `from logger import log` 直接使用
log = setup_logging()
