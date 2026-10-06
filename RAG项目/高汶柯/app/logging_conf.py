"""统一日志配置：控制台 + 滚动文件，按环境分级。"""
import logging
import sys
from logging.handlers import RotatingFileHandler

from app.config import settings

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def setup_logging() -> logging.Logger:
    root = logging.getLogger()
    if root.handlers:  # 避免重复初始化
        return logging.getLogger("rag")

    root.setLevel(logging.WARNING if settings.is_prod else logging.INFO)
    formatter = logging.Formatter(_LOG_FORMAT)

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    root.addHandler(stream)

    file_handler = RotatingFileHandler(
        "rag.log", maxBytes=10 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    return logging.getLogger("rag")


log = setup_logging()
