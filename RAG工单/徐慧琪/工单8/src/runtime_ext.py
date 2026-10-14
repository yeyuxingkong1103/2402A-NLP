# 工单编号：人工智能NLP-RAG-基于GraphRAG实现金融问答
"""
模块：容错与长时间运行支撑（硬性要求7）

提供三件套：
  1. setup_logging —— 按模块的**日志轮转**（单文件上限 + 备份数），
     长时间跑批不会把磁盘写满；
  2. MemoryMonitor —— 进程内存周期采样，超阈值告警；
  3. guarded —— 把易失败操作（解压/解析/抽取/查询/模型加载/LLM 调用）
     包成「失败即降级并留痕」，对应工单要求的容错清单。
"""
from __future__ import annotations

import functools
import logging
import logging.handlers
import os
import threading
import time

from src import config

LOG_MAX_BYTES = 8 * 1024 * 1024          # 单文件 8MB
LOG_BACKUP_COUNT = 5                     # 保留 5 份历史
LOG_FORMAT = "%(asctime)s %(levelname)-7s [%(name)s] %(message)s"


def setup_logging(name: str, level: int = logging.INFO) -> logging.Logger:
    """建/取一个带轮转文件 + 控制台输出的 logger（幂等）。"""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if getattr(logger, "_wo08_configured", False):
        return logger

    os.makedirs(config.LOG_DIR, exist_ok=True)
    fmt = logging.Formatter(LOG_FORMAT)

    fh = logging.handlers.RotatingFileHandler(
        os.path.join(config.LOG_DIR, f"{name}.log"),
        maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUP_COUNT, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    logger._wo08_configured = True          # type: ignore[attr-defined]
    return logger


class MemoryMonitor:
    """进程内存周期采样（后台线程）。"""

    def __init__(self, threshold_mb: float = 6000.0,
                 interval_s: float = 30.0) -> None:
        self.threshold_mb = threshold_mb
        self.interval_s = interval_s
        self._peak = 0.0
        self._current = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._log = setup_logging("memory")

    def _sample(self) -> float:
        try:
            import psutil
            mb = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        except Exception:                       # noqa: BLE001
            return 0.0
        self._current = mb
        self._peak = max(self._peak, mb)
        if mb > self.threshold_mb:
            self._log.warning("内存占用 %.0f MB 超过阈值 %.0f MB",
                              mb, self.threshold_mb)
        return mb

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            self._sample()

    def start(self) -> "MemoryMonitor":
        self._sample()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="wo08-memmon")
        self._thread.start()
        return self

    def stop(self) -> dict:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._sample()
        return self.stats()

    def stats(self) -> dict:
        return {"current_mb": round(self._current, 1),
                "peak_mb": round(self._peak, 1),
                "threshold_mb": self.threshold_mb,
                "over_threshold": self._peak > self.threshold_mb}


def guarded(op_name: str, fallback=None, reraise: bool = False):
    """装饰器：把易失败操作包成「失败即降级并留痕」。

    对应工单容错清单中的：zip 解压 / PDF 解析 / 实体识别 / 关系抽取 /
    图库读写 / 图谱查询 / 模型加载 / LLM 调用。
    """
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            log = setup_logging("fault")
            try:
                return fn(*args, **kwargs)
            except Exception as exc:            # noqa: BLE001
                log.error("[%s] 失败并降级：%s: %s",
                          op_name, type(exc).__name__, exc)
                if reraise:
                    raise
                return fallback
        return wrapper
    return deco


def health_check() -> dict:
    """汇总各子系统健康状态（供界面「系统状态」页与 CLI 使用）。"""
    out: dict = {"ok": True, "graph": {}, "llm": {}, "retriever": {}}

    try:
        from src import graph_qa
        out["graph"] = graph_qa.status()
        if not out["graph"].get("graph_loaded"):
            out["ok"] = False
    except Exception as exc:                    # noqa: BLE001
        out["graph"] = {"error": str(exc)}
        out["ok"] = False

    try:
        from src import llm
        out["llm"] = llm.health()
        if not out["llm"].get("ok"):
            out["ok"] = False
    except Exception as exc:                    # noqa: BLE001
        out["llm"] = {"ok": False, "error": str(exc)}
        out["ok"] = False

    try:
        from src import retriever, strategies
        out["retriever"] = {"stats": retriever.stats(),
                            "strategies": strategies.describe()}
    except Exception as exc:                    # noqa: BLE001
        out["retriever"] = {"error": str(exc)}
        out["ok"] = False

    return out
