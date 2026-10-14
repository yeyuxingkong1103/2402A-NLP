# 工单编号：人工智能NLP-RAG-基于GraphRAG实现金融问答
import logging
import os
import time

from src.runtime_ext import (MemoryMonitor, guarded, health_check,
                             setup_logging)


def test_setup_logging_creates_rotating_file(tmp_path, monkeypatch):
    from src import config
    monkeypatch.setattr(config, "LOG_DIR", str(tmp_path))
    logger = setup_logging("test_mod")
    logger.info("hello")
    files = os.listdir(tmp_path)
    assert any(f.endswith(".log") for f in files), files
    # 轮转 handler 必须配置
    handlers = [h for h in logger.handlers
                if isinstance(h, logging.handlers.RotatingFileHandler)]
    assert handlers, "必须使用 RotatingFileHandler（长时间运行要求）"
    assert handlers[0].maxBytes > 0
    assert handlers[0].backupCount >= 1


def test_setup_logging_is_idempotent(tmp_path, monkeypatch):
    from src import config
    monkeypatch.setattr(config, "LOG_DIR", str(tmp_path))
    a = setup_logging("dup_mod")
    n = len(a.handlers)
    b = setup_logging("dup_mod")
    assert len(b.handlers) == n, "重复调用不应叠加 handler"


def test_guarded_returns_result():
    @guarded("op_ok")
    def ok():
        return 42
    assert ok() == 42


def test_guarded_returns_fallback_on_error():
    @guarded("op_fail", fallback="降级值")
    def boom():
        raise RuntimeError("故意的")
    assert boom() == "降级值"


def test_guarded_reraise_when_requested():
    @guarded("op_reraise", reraise=True)
    def boom():
        raise ValueError("必须抛出")
    try:
        boom()
    except ValueError:
        pass
    else:
        raise AssertionError("应当抛出 ValueError")


def test_memory_monitor_stats_shape():
    mon = MemoryMonitor(threshold_mb=99999, interval_s=0.1)
    mon.start()
    time.sleep(0.25)
    st = mon.stop()
    assert "current_mb" in st
    assert "peak_mb" in st
    assert "threshold_mb" in st
    assert "over_threshold" in st
    assert st["peak_mb"] > 0


def test_health_check_returns_subsystems():
    h = health_check()
    for key in ("graph", "llm", "retriever"):
        assert key in h
    assert "ok" in h
