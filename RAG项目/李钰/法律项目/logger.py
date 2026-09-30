# -*- coding: utf-8 -*-
"""
统一日志模块
- 一次性配置根 logger：控制台 + 按大小轮转的文件日志
- 各业务模块通过 `from logger import get_logger` 获取带名字的 logger
- 不影响未使用本模块的第三方库；只对 root 配置 handler，避免重复输出
"""
import logging
import logging.handlers
import threading
import sys
from pathlib import Path

import config as C

_CONFIGURED = False
_LOCK = threading.Lock()


def _level_from_str(s, default=logging.INFO):
    """把字符串级别安全地转成 logging 常量"""
    try:
        return int(getattr(logging, str(s).upper())) if isinstance(s, str) else int(s)
    except Exception:
        return default


def configure(force: bool = False) -> None:
    """初始化根 logger 的 handler/format/级别；多次调用幂等（除非 force=True）"""
    global _CONFIGURED
    with _LOCK:
        if _CONFIGURED and not force:
            return

        log_dir = Path(getattr(C, "LOG_DIR", Path(__file__).parent / "logs"))
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = Path(getattr(C, "LOG_FILE", log_dir / "app.log"))

        fmt = logging.Formatter(
            fmt=getattr(C, "LOG_FORMAT",
                        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"),
            datefmt=getattr(C, "LOG_DATE_FMT", "%Y-%m-%d %H:%M:%S"),
        )

        root = logging.getLogger()
        level = _level_from_str(getattr(C, "LOG_LEVEL", "INFO"))
        root.setLevel(level)

        # 清理已有 handler，避免重复输出（如重复 import 本模块/单元测试）
        if force:
            for h in list(root.handlers):
                root.removeHandler(h)

        # 控制台 handler
        has_stream = any(isinstance(h, logging.StreamHandler)
                         and not isinstance(h, logging.handlers.RotatingFileHandler)
                         for h in root.handlers)
        if not has_stream:
            sh = logging.StreamHandler(stream=sys.stdout)
            sh.setFormatter(fmt)
            sh.setLevel(level)
            root.addHandler(sh)

        # 文件 handler（按大小轮转，保留 N 个备份）
        has_file = any(isinstance(h, logging.handlers.RotatingFileHandler)
                      for h in root.handlers)
        if not has_file:
            fh = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=int(getattr(C, "LOG_FILE_MAX_BYTES", 5 * 1024 * 1024)),
                backupCount=int(getattr(C, "LOG_FILE_BACKUP_COUNT", 5)),
                encoding="utf-8",
                delay=True,
            )
            fh.setFormatter(fmt)
            fh.setLevel(level)
            root.addHandler(fh)

        _CONFIGURED = True
        # 用 root 自身输出一条启动信息，便于在日志里看到初始化时间点
        logging.getLogger("logger").info(
            "日志系统已初始化 | level=%s | file=%s", logging.getLevelName(level), log_file)


def get_logger(name: str = "app") -> logging.Logger:
    """获取一个命名 logger；首次调用时自动配置根 logger"""
    if not _CONFIGURED:
        configure()
    return logging.getLogger(name)


def tail_logs(n: int = None, level: str = None):
    """读取主日志文件末尾 n 行（默认 LOG_TAIL_DEFAULT）；可选级别过滤"""
    n = n or int(getattr(C, "LOG_TAIL_DEFAULT", 200))
    n = max(1, min(int(n), 5000))
    log_file = Path(getattr(C, "LOG_FILE", C.LOG_DIR / "app.log"))
    if not log_file.exists():
        return []
    # 用 deque 高效 tail，避免一次性读取大文件
    from collections import deque
    buf = deque(maxlen=n)
    try:
        with open(log_file, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                buf.append(line.rstrip("\r\n"))
    except Exception as e:
        return [f"[logger] 读取日志失败: {e}"]
    lines = list(buf)
    if level:
        lv = str(level).upper()
        # 仅按行内包含的级别字样过滤（轻量、不依赖解析）
        lines = [ln for ln in lines if f"| {lv}" in ln or f"| {lv:7s}" in ln]
    return lines


# 模块导入时即完成一次配置，确保任何模块 `from logger import get_logger` 之后立即可用
configure()
