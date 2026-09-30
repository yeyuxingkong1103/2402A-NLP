# -*- coding: utf-8 -*-
"""
日志模块：统一使用 Python logging，控制台 + 文件双输出。

在系统中的位置：
    被 db_redis / db_mysql / db_milvus / retriever / main 等模块 import 后，
    调用 get_logger(__name__) 拿到属于自己模块的 logger；本模块只依赖 config，
    不依赖任何业务模块，处于依赖链的最底层。

职责：
    给每个模块一个「带模块名前缀」的 logger，同时往控制台和轮转文件输出，
    这样线上排查时既能实时看终端，又能事后翻日志文件对账。

关键设计取舍：
    1. 用 logging.getLogger(name) 而不是自己造对象：Python 的 logger 是按名字
       全局注册的，同名 getLogger 天然返回同一个实例，等于免费得到了单例；
    2. 用 if logger.handlers 做幂等保护：模块被重复 import 或本函数被多次调用时，
       避免同一个 logger 挂上多个 handler（否则一条日志会被打印 N 遍）；
    3. 文件用 RotatingFileHandler 而不是普通 FileHandler：单文件写到 10MB 就切分、
       最多保留 3 个备份，防止服务长期运行把磁盘写满。
"""

import logging  # Python 标准日志库
import os  # 路径拼接
from logging.handlers import RotatingFileHandler  # 日志轮转：文件过大自动切分

from config import LOG_LEVEL, LOG_FILE  # 从配置读取日志级别和文件路径


def get_logger(name: str) -> logging.Logger:
    """
    创建/获取一个 logger：控制台输出 + 文件轮转输出。

    参数：
        name：logger 名称，调用方统一传 __name__（即模块名，如 "db_milvus"），
              日志行中会以 [模块名] 的形式出现，便于定位日志来源。
    返回：
        logging.Logger 实例；同名多次调用返回同一个对象（已初始化过则直接返回）。
    异常与降级：
        日志文件所在目录不存在或无写权限时，RotatingFileHandler 在构造阶段就会抛
        OSError。本函数刻意不捕获——启动阶段直接失败，比「日志静默丢失」更容易被
        发现和定位。
    """
    logger = logging.getLogger(name)  # 按模块名创建 logger（重复调用返回同一个）
    if logger.handlers:  # 已经初始化过（避免重复添加 handler）
        return logger  # 直接返回；少了这段保护，重复调用会让同一条日志打印多次
    logger.setLevel(LOG_LEVEL)  # 设置最低日志级别（低于该级别的日志直接丢弃，不产生 I/O 开销）
    fmt = logging.Formatter(  # 日志格式
        "%(asctime)s [%(levelname)s] [%(name)s] %(message)s",  # 时间 级别 模块名 消息
        datefmt="%Y-%m-%d %H:%M:%S",  # 时间格式（人类可读，不带毫秒）
    )
    # 控制台输出
    console = logging.StreamHandler()  # 控制台 handler（默认写到 stderr，不会和业务 print 抢 stdout 管道）
    console.setFormatter(fmt)  # 设置格式
    logger.addHandler(console)  # 添加到 logger
    # 文件输出（轮转：单文件 10MB，保留 3 个备份）
    file_handler = RotatingFileHandler(  # 文件 handler
        LOG_FILE,  # 日志文件路径（相对路径时基于进程启动目录）
        maxBytes=10 * 1024 * 1024,  # 单文件最大 10MB
        backupCount=3,  # 保留 3 个历史文件（persona_rag.log.1 / .2 / .3）
        encoding="utf-8",  # UTF-8 编码（中文不乱码；Windows 默认 GBK 会乱码甚至写入报错）
    )
    file_handler.setFormatter(fmt)  # 设置格式
    logger.addHandler(file_handler)  # 添加到 logger
    return logger  # 返回配置好的 logger