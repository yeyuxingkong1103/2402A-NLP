# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG项目-LightRAG 优化任务
# 模块职责：提供全局统一的日志对象，供各模块复用，保证输出格式一致且带时间戳

import logging                                                  # 导入标准日志库
import config                                                   # 导入本工单配置模块（顺带修正标准输出编码）

_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"             # 日志格式：时间戳 + 级别 + 内容
_DATEFMT = "%H:%M:%S"                                           # 时间戳格式：仅到秒，保持输出紧凑


def get_logger(name="lightrag"):                                # 定义日志对象获取函数
    """返回一个配置好的日志对象；重复调用不会重复添加处理器。"""
    logger = logging.getLogger(name)                            # 按名称取出（或创建）日志对象
    if logger.handlers:                                         # 已配置过则直接返回，避免日志重复打印
        return logger                                           # 返回现有对象
    handler = logging.StreamHandler()                           # 创建输出到控制台的处理器
    handler.setFormatter(logging.Formatter(_FORMAT, _DATEFMT))  # 设置统一的格式
    logger.addHandler(handler)                                  # 挂载处理器
    logger.setLevel(logging.INFO)                               # 默认输出 INFO 及以上级别
    logger.propagate = False                                    # 阻止向根日志传播，避免重复输出
    return logger                                               # 返回配置好的日志对象


LOGGER = get_logger()                                           # 模块级单例，供各模块直接导入使用
