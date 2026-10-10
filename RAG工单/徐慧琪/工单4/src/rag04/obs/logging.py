# 工单编号：人工智能NLP-RAG-图像内容解析及检索优化
"""日志轮转 + 计时 + 内存快照，满足硬性要求 7「长时间运行」。"""
from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from logging.handlers import RotatingFileHandler
from typing import Iterator

from rag04.config import Settings

_FMT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def setup_logging(settings: Settings, name: str = "rag04") -> logging.Logger:
    """配置轮转日志（10MB × 10 份）。重复调用幂等。"""
    settings.log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not any(isinstance(h, RotatingFileHandler) for h in logger.handlers):
        fh = RotatingFileHandler(
            settings.log_dir / f"{name}.log",
            maxBytes=10 * 1024 * 1024,
            backupCount=10,
            encoding="utf-8",
        )
        fh.setFormatter(logging.Formatter(_FMT))
        logger.addHandler(fh)

    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, RotatingFileHandler)
               for h in logger.handlers):
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter(_FMT))
        logger.addHandler(sh)

    return logger


@contextmanager
def timed(logger: logging.Logger, label: str) -> Iterator[dict]:
    """计时上下文：退出时记录耗时并回填到 yield 出的 dict。"""
    rec: dict = {"label": label}
    t0 = time.perf_counter()
    try:
        yield rec
    finally:
        rec["elapsed_ms"] = (time.perf_counter() - t0) * 1000.0
        logger.info("%s 耗时 %.1f ms", label, rec["elapsed_ms"])


def mem_snapshot() -> dict:
    """返回当前进程 RSS(MB) 与可用时的 GPU 显存占用(MB)。

    ``rss_mb == 0.0`` 表示**未测得**（psutil 与 POSIX ``resource`` 都不可用），
    不再用正数占位：旧实现填 1.0 兜底，会被压测报告当成真实 RSS 打印出来。
    ``gpu_mb is None`` 同样是未测得。消费方（``benchmarks/loadtest.py``）据此
    渲染「未测得」，不把缺失读数伪装成数字。
    """
    rss_mb = 0.0                                   # 0.0 = 未测得
    try:
        import psutil  # 主选：跨平台
        rss_mb = psutil.Process().memory_info().rss / 1e6
    except Exception:
        try:
            import resource  # 备选：POSIX only
            rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        except Exception:
            rss_mb = 0.0                           # 测不到就是 0.0，绝不编造正数
    gpu_mb = None
    try:
        import torch
        if torch.cuda.is_available():
            gpu_mb = torch.cuda.memory_allocated() / 1e6
    except Exception:
        pass
    return {"rss_mb": rss_mb, "gpu_mb": gpu_mb}
