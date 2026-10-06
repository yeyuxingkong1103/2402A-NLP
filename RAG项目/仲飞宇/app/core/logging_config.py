"""统一的 python logging 配置：控制台 + 按天滚动文件。"""
from __future__ import annotations

import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

from .config import Settings

_CONFIGURED = False


def setup_logging(settings: Settings) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    log_dir = Path(settings.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(settings.log_level.upper())

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 控制台
    console = logging.StreamHandler()
    console.setLevel(settings.log_level.upper())
    console.setFormatter(fmt)

    # 文件（按天滚动，保留 7 天）。
    #
    # 文件名按进程分：多 worker 共写一个 app.log 时，TimedRotatingFileHandler 的轮转是
    # 「把自己那个文件改名成带日期的 → 再新建同名文件」，三个进程各持自己的 fd，谁先跨天
    # 谁改名，另外两个的 fd 就指向了被改名的文件。实测（两进程共用 LOG_DIR，真实
    # setup_logging，只把周期换成 1 秒以便观察）跨过轮转点后：
    #   app.log                     -> 只剩"赢的"那个进程的 1 行
    #   app.log.2026-09-21_19-54-19 -> 两个进程轮转前的 2 行
    #   "输的"进程轮转后那行        -> FileNotFoundError in doRollover，**这行日志直接丢**
    # 于是当天 app.log 只有一个 worker 在写，排查时会得出"另外两个 worker 没打日志"的错误
    # 结论（而日志口径正是本项目最在意的那类坑）。按端口分文件后每个文件只有一个写者，
    # 轮转互不干扰；单实例（WORKER_PORT 为空）仍写 logs/app.log，部署文档与自检脚本引用
    # 的就是这个路径。
    log_file = f"app-{settings.worker_port}.log" if settings.worker_port else "app.log"
    file_handler = TimedRotatingFileHandler(
        log_dir / log_file, when="midnight", backupCount=7, encoding="utf-8"
    )
    file_handler.setLevel(settings.log_level.upper())
    file_handler.setFormatter(fmt)

    root.handlers = [console, file_handler]

    # 降低第三方库日志噪音
    for noisy in ("uvicorn.access", "pymilvus", "httpx", "httpcore", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
