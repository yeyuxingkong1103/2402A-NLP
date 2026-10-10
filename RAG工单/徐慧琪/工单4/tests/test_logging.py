# 工单编号：人工智能NLP-RAG-图像内容解析及检索优化
import logging
from rag04.config import get_settings
from rag04.obs.logging import setup_logging, timed, mem_snapshot


def test_setup_logging_writes_rotating_file(tmp_path):
    s = get_settings()
    object.__setattr__(s, "log_dir", tmp_path)
    logger = setup_logging(s, name="rag04_test")
    logger.info("hello 工单04")
    for h in logger.handlers:
        h.flush()
    files = list(tmp_path.glob("*.log"))
    assert files, "应生成日志文件"
    assert "工单04" in files[0].read_text(encoding="utf-8")


def test_logging_has_rotation_handler(tmp_path):
    from logging.handlers import RotatingFileHandler
    s = get_settings()
    object.__setattr__(s, "log_dir", tmp_path)
    logger = setup_logging(s, name="rag04_rot")
    rot = [h for h in logger.handlers if isinstance(h, RotatingFileHandler)]
    assert rot, "必须配置日志轮转"
    assert rot[0].maxBytes == 10 * 1024 * 1024
    assert rot[0].backupCount == 10


def test_timed_records_elapsed():
    logger = logging.getLogger("rag04_time")
    with timed(logger, "unit") as rec:
        pass
    assert "elapsed_ms" in rec
    assert rec["elapsed_ms"] >= 0


def test_mem_snapshot_has_rss():
    snap = mem_snapshot()
    assert snap["rss_mb"] > 0
