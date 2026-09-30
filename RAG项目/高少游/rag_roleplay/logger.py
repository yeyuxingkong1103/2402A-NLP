# -*- coding: utf-8 -*-
"""
日志模块：统一使用 Python logging，控制台 + 文件双输出
所有模块通过 get_logger() 获取 logger，自动带模块名前缀
"""

import logging  # Python 标准日志库
import os  # 路径拼接
from logging.handlers import RotatingFileHandler  # 日志轮转：文件过大自动切分

from config import LOG_LEVEL, LOG_FILE  # 从配置读取日志级别和文件路径


def get_logger(name: str) -> logging.Logger:
    """创建/获取一个 logger：控制台输出 + 文件轮转输出"""
    logger = logging.getLogger(name)  # 按模块名创建 logger（重复调用返回同一个）
    if logger.handlers:  # 已经初始化过（避免重复添加 handler）
        return logger  # 直接返回
    logger.setLevel(LOG_LEVEL)  # 设置最低日志级别
    fmt = logging.Formatter(  # 日志格式
        "%(asctime)s [%(levelname)s] [%(name)s] %(message)s",  # 时间 级别 模块名 消息
        datefmt="%Y-%m-%d %H:%M:%S",  # 时间格式
    )
    # 控制台输出
    console = logging.StreamHandler()  # 控制台 handler
    console.setFormatter(fmt)  # 设置格式
    logger.addHandler(console)  # 添加到 logger
    # 文件输出（轮转：单文件 10MB，保留 3 个备份）
    file_handler = RotatingFileHandler(  # 文件 handler
        LOG_FILE,  # 日志文件路径
        maxBytes=10 * 1024 * 1024,  # 单文件最大 10MB
        backupCount=3,  # 保留 3 个历史文件
        encoding="utf-8",  # UTF-8 编码（中文不乱码）
    )
    file_handler.setFormatter(fmt)  # 设置格式
    logger.addHandler(file_handler)  # 添加到 logger
    return logger  # 返回配置好的 logger
