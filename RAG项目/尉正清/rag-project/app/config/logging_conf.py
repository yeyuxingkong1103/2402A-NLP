# app/config/logging_conf.py
"""日志配置：使用 Python 标准库 logging。

三路输出，各司其职：
    控制台    开发时看，带级别配色，只输出 INFO 以上
    应用日志  logs/app.log，按大小滚动，保留 5 份历史
    错误日志  logs/error.log，只收 WARNING 以上，便于出问题时快速定位

级别可用环境变量 LOG_LEVEL 覆盖（DEBUG / INFO / WARNING / ERROR）。
"""
import logging.handlers
import os
import sys

from app.config import settings
import logging

logger = logging.getLogger(__name__)

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-28s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
MAX_BYTES = 10 * 1024 * 1024        # 单文件 10MB
BACKUP_COUNT = 5

# 这些库的日志在 INFO 级别过于聒噪，压到 WARNING
NOISY_LOGGERS = {
    "httpx": logging.WARNING,
    "httpcore": logging.WARNING,
    "urllib3": logging.WARNING,
    "openai": logging.WARNING,
    "pymilvus": logging.WARNING,
    "milvus_lite": logging.WARNING,
    "mysql.connector": logging.WARNING,
    "sentence_transformers": logging.WARNING,
    "transformers": logging.WARNING,
    "datasets": logging.WARNING,
    "matplotlib": logging.WARNING,
    "PIL": logging.WARNING,
}

_configured = False


class _ColorFormatter(logging.Formatter):
    """控制台配色：ERROR 红、WARNING 黄、INFO 默认、DEBUG 灰。"""

    COLORS = {
        logging.DEBUG: "\033[90m",
        logging.INFO: "\033[0m",
        logging.WARNING: "\033[33m",
        logging.ERROR: "\033[31m",
        logging.CRITICAL: "\033[41m",
    }
    RESET = "\033[0m"

    def __init__(self, use_color=True):
        super().__init__(LOG_FORMAT, DATE_FORMAT)
        self.use_color = use_color and sys.stderr.isatty()

    def format(self, record):
        text = super().format(record)
        if not self.use_color:
            return text
        return "%s%s%s" % (self.COLORS.get(record.levelno, ""), text, self.RESET)


def setup_logging(level=None, log_dir=None, force=False):
    """初始化根 logger。重复调用是安全的（除非 force=True）。"""
    global _configured
    if _configured and not force:
        return logging.getLogger()

    level = (level or os.getenv("LOG_LEVEL") or "INFO").upper()
    log_dir = log_dir or settings.LOGS_DIR
    os.makedirs(log_dir, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(getattr(logging, level, logging.INFO))
    for h in list(root.handlers):       # 清掉可能已存在的 handler，避免重复输出
        root.removeHandler(h)

    # 控制台
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(logging.DEBUG if level == "DEBUG" else logging.INFO)
    console.setFormatter(_ColorFormatter())
    root.addHandler(console)

    # 应用日志（滚动）
    app_handler = logging.handlers.RotatingFileHandler(
        os.path.join(log_dir, "app.log"), maxBytes=MAX_BYTES,
        backupCount=BACKUP_COUNT, encoding="utf-8")
    app_handler.setLevel(logging.DEBUG)
    app_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
    root.addHandler(app_handler)

    # 错误日志
    err_handler = logging.handlers.RotatingFileHandler(
        os.path.join(log_dir, "error.log"), maxBytes=MAX_BYTES,
        backupCount=BACKUP_COUNT, encoding="utf-8")
    err_handler.setLevel(logging.WARNING)
    err_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
    root.addHandler(err_handler)

    for name, lvl in NOISY_LOGGERS.items():
        logging.getLogger(name).setLevel(lvl)

    _configured = True
    logging.getLogger(__name__).debug("日志已初始化 level=%s dir=%s", level, log_dir)
    return root
