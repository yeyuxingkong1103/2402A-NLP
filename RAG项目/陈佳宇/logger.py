import logging
import os
from datetime import datetime

# 日志目录
LOG_DIR = "logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# 日志文件名按日期
log_file = os.path.join(LOG_DIR, f"rag_{datetime.now().strftime('%Y%m%d')}.log")

# 配置logger
logger = logging.getLogger("rag_character")
logger.setLevel(logging.DEBUG)

# 避免重复添加handler
if not logger.handlers:
    # 文件handler：记录所有级别
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(filename)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_fmt)
    logger.addHandler(file_handler)

    # 控制台handler：只记录INFO及以上
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")
    console_handler.setFormatter(console_fmt)
    logger.addHandler(console_handler)

def get_logger():
    return logger
