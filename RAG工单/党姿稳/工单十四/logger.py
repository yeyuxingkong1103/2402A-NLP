# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-14-修复低质量工业PDF的解析与信息丢失
# 模块职责：统一日志出口，同名 logger 复用，避免重复挂载 Handler

import logging                                                   # 标准日志库
import sys                                                       # 用于指定输出流

_LOGGERS = {}                                                    # 已创建的 logger 缓存（按名称复用）

_FORMAT = "%(asctime)s [%(levelname)s] %(name)s | %(message)s"    # 统一日志格式
_DATEFMT = "%H:%M:%S"                                            # 只保留时分秒，控制台更紧凑


def get_logger(name="rag14", level=logging.INFO):
    """返回带控制台 Handler 的 logger，同名重复调用返回同一实例。"""
    if name in _LOGGERS:                                         # 已创建过直接复用，避免日志重复打印
        return _LOGGERS[name]
    logger = logging.getLogger(name)                             # 取（或创建）logger
    logger.setLevel(level)                                       # 设置级别
    logger.propagate = False                                     # 不向 root 传播，防止重复输出
    handler = logging.StreamHandler(sys.stdout)                  # 输出到标准输出
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
    logger.addHandler(handler)
    _LOGGERS[name] = logger
    return logger


def banner(logger, text, width=78):
    """打印醒目的分隔标题，用于标记流水线阶段。"""
    line = "=" * width                                           # 顶部分隔线
    logger.info(line)
    logger.info(text)
    logger.info(line)
