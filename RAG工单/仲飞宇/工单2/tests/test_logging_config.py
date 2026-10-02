"""日志落盘的文件名口径。

只钉住"哪个进程写哪个文件"这一条：多 worker 共写一个 app.log 时，
TimedRotatingFileHandler 的轮转（改名 + 新建同名文件）会互相踩。
"""
from __future__ import annotations

import logging
from logging.handlers import TimedRotatingFileHandler

import pytest

from app.core import logging_config
from app.core.config import Settings


@pytest.fixture
def restore_logging():
    """setup_logging 会改 root logger 的 handlers（还给模块级 _CONFIGURED 置位），用完还原。"""
    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_flag = logging_config._CONFIGURED
    yield
    for h in list(root.handlers):
        if h not in saved_handlers:
            h.close()
    root.handlers = saved_handlers
    logging_config._CONFIGURED = saved_flag


def _file_handler_path():
    for h in logging.getLogger().handlers:
        if isinstance(h, TimedRotatingFileHandler):
            return h.baseFilename
    raise AssertionError("没有文件 handler")


def test_worker_port_gets_its_own_log_file(tmp_path, restore_logging):
    """多 worker：日志按端口分文件，跨进程轮转的共享点就不存在了。

    实测两进程共写一个 app.log（真实 setup_logging，只把轮转周期换成 1 秒）：跨过轮转点后
    app.log 只剩"赢的"那个进程一行，另一个进程"轮转后"那行在 doRollover 里抛
    FileNotFoundError 直接丢——当天日志看起来就是"另一个 worker 没打日志"。
    """
    logging_config._CONFIGURED = False
    logging_config.setup_logging(Settings(log_dir=str(tmp_path), worker_port="8002"))

    assert _file_handler_path() == str(tmp_path / "app-8002.log")


def test_single_instance_keeps_app_log(tmp_path, restore_logging):
    """单实例（WORKER_PORT 为空）路径不变：部署文档与自检脚本引用的就是 logs/app.log。"""
    logging_config._CONFIGURED = False
    logging_config.setup_logging(Settings(log_dir=str(tmp_path)))

    assert _file_handler_path() == str(tmp_path / "app.log")
