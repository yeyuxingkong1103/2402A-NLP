# -*- coding: utf-8 -*-
"""日志模块：只负责 Python logging 的基础配置，第 0 步不包含业务逻辑。"""

import logging                 # 导入 logging 模块，用于记录日志
from pathlib import Path       # 导入 Path 类，用于拼接日志文件路径

BASE_DIR = Path(__file__).resolve().parent      # 项目根目录
LOG_DIR = BASE_DIR / "logs"                     # 日志文件存放目录
LOG_FILE = LOG_DIR / "app.log"                  # 日志文件完整路径
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"  # 日志输出格式
LOG_LEVEL = logging.INFO                        # 默认日志级别为 INFO

_LOGGER_CACHE = {}  # 缓存已创建的 logger，避免重复添加处理器


def get_logger(name: str = "rag") -> logging.Logger:
    """获取指定名称的 logger；控制台与文件同时输出，重复调用不会重复加处理器。"""
    if name in _LOGGER_CACHE:            # 如果该名称的 logger 已经创建过
        return _LOGGER_CACHE[name]       # 直接返回缓存中的 logger
    logger = logging.getLogger(name)     # 按名称获取 logger 实例
    logger.setLevel(LOG_LEVEL)           # 设置日志级别
    logger.propagate = False             # 关闭向上级 logger 传递，避免日志重复打印
    if not logger.handlers:              # 如果还没有添加过任何处理器
        console_handler = logging.StreamHandler()          # 创建控制台输出处理器
        console_handler.setFormatter(logging.Formatter(LOG_FORMAT))  # 设置控制台格式
        logger.addHandler(console_handler)                 # 把控制台处理器挂到 logger 上
        try:                                               # 尝试添加文件处理器（可能无写权限）
            LOG_DIR.mkdir(parents=True, exist_ok=True)     # 确保日志目录存在
            file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")  # 创建文件输出处理器
            file_handler.setFormatter(logging.Formatter(LOG_FORMAT))       # 设置文件格式
            logger.addHandler(file_handler)                # 把文件处理器挂到 logger 上
        except OSError:                                    # 目录或文件创建失败时
            logger.warning("日志文件初始化失败，仅输出到控制台")  # 记录一条告警并继续
    _LOGGER_CACHE[name] = logger         # 把创建好的 logger 存入缓存
    return logger                        # 返回 logger 实例
