# -*- coding: utf-8 -*-
"""统一日志配置：控制台 + 按大小轮转的文件。"""
import logging
import sys
from logging.handlers import RotatingFileHandler

from . import config

_configured = False

FMT = "%(asctime)s | %(levelname)-7s | %(name)-18s | %(message)s"


def setup_logging(level: int = logging.INFO) -> None:
    global _configured
    if _configured:
        return

    root = logging.getLogger()
    root.setLevel(level)
    formatter = logging.Formatter(FMT, datefmt="%Y-%m-%d %H:%M:%S")

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    root.addHandler(console)

    file_handler = RotatingFileHandler(
        config.LOG_DIR / "app.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    # 第三方库降噪
    for noisy in ("urllib3", "httpx", "httpcore", "transformers", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)
