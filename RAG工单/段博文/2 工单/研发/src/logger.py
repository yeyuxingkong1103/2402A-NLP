# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化
"""
日志模块：统一 logger 工厂，控制台 + 文件双输出。
"""

import logging
import os
from logging.handlers import RotatingFileHandler

from config import LOG_LEVEL, LOG_FILE

_CONFIGURED = False


def _configure_root():
    global _CONFIGURED
    if _CONFIGURED:
        return
    level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    root.addHandler(sh)

    log_path = os.path.abspath(LOG_FILE)
    fh = RotatingFileHandler(log_path, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    _configure_root()
    return logging.getLogger(name)
