# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-金融问答系统部署
# 模块职责：统一日志配置——同时输出到控制台与日志文件，满足容器日志与卷持久化要求

import os                                                       # 导入 os 用于日志目录处理
import sys                                                      # 导入 sys 用于指定输出流
import logging                                                  # 导入标准日志库
from logging.handlers import RotatingFileHandler                # 导入按大小滚动的文件处理器
import config                                                   # 导入本工单配置模块

LOGGER = logging.getLogger("rag_service")                       # 创建服务专用 logger
LOGGER.setLevel(logging.INFO)                                   # 设置日志级别为 INFO
LOGGER.propagate = False                                        # 禁止向上传播，避免根 logger 重复输出

_FORMAT = "%(asctime)s [%(levelname)s] %(name)s | %(message)s"  # 日志格式：时间 级别 名称 | 内容
_INITED = False                                                 # 初始化标记，防止重复添加处理器


def init_logging():                                             # 定义日志初始化函数
    """初始化控制台与文件日志处理器；重复调用只生效一次。"""
    global _INITED                                              # 声明模块级标记
    if _INITED:                                                 # 已初始化则直接返回
        return LOGGER                                           # 返回 logger
    config.ensure_dirs()                                        # 确保日志目录存在
    fmt = logging.Formatter(_FORMAT)                            # 创建格式化器
    console = logging.StreamHandler(sys.stdout)                 # 控制台处理器（输出到标准输出便于 docker logs 查看）
    console.setFormatter(fmt)                                   # 应用格式
    LOGGER.addHandler(console)                                  # 注册控制台处理器
    try:                                                        # 尝试创建文件处理器（验证卷挂载）
        file_handler = RotatingFileHandler(                     # 按大小滚动的文件处理器
            os.path.join(config.LOG_DIR, "service.log"),        # 日志文件路径（位于挂载卷内）
            maxBytes=5 * 1024 * 1024,                           # 单文件上限 5MB
            backupCount=3,                                      # 最多保留 3 个历史文件
            encoding="utf-8")                                   # 使用 UTF-8 编码
        file_handler.setFormatter(fmt)                          # 应用格式
        LOGGER.addHandler(file_handler)                         # 注册文件处理器
    except Exception as e:                                      # 目录不可写等异常
        LOGGER.warning("文件日志不可用，仅输出到控制台：%r", e)   # 记录告警但不中断服务
    _INITED = True                                              # 标记已完成初始化
    return LOGGER                                               # 返回 logger


def log_startup_banner():                                       # 定义启动横幅日志函数
    """输出启动配置摘要，便于排查部署问题。"""
    logger = init_logging()                                     # 确保日志已初始化
    logger.info("=" * 60)                                       # 输出分隔线
    logger.info("金融问答服务启动中，版本 %s", config.APP_VERSION)  # 输出版本信息
    logger.info("数据目录：%s", config.DATA_DIR)                 # 输出数据目录（应为挂载卷）
    logger.info("日志目录：%s", config.LOG_DIR)                  # 输出日志目录（应为挂载卷）
    logger.info("语料目录：%s", config.CORPUS_DIR)               # 输出语料目录
    logger.info("模型路径：%s（向量检索：%s）",                    # 输出模型路径
                config.MODEL_PATH, "开启" if config.EMBED_ENABLED else "关闭")  # 输出向量检索开关
    logger.info("服务端口：%d", config.SERVICE_PORT)             # 输出监听端口
    logger.info("=" * 60)                                       # 输出分隔线
