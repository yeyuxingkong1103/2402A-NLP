# -*- coding: utf-8 -*-
"""
日志配置模块

职责边界（对应需求文档 N5 可观测性）：
    本模块负责把**检索中间过程**完整记录到服务端日志，用于开发调试与评测排查。
    这些 Trace 信息**仅保存在服务端日志文件中**：
        - 不通过 HTTP 响应返回给前端
        - 不提供任何 Trace 查询接口
        - 前端页面不展示检索链路细节

日志内容分三类：
    1. 服务请求   —— 接口路径、请求 ID、会话 ID、耗时、状态码
    2. 检索过程   —— 用户原始问题、命中块、相似度分数、命中页码、上下文长度
    3. 错误堆栈   —— 完整异常堆栈、失败环节、入参快照
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
import uuid
from contextvars import ContextVar
from typing import Optional

from backend.config import settings

# ---------------------------------------------------------------------------
# 请求上下文：让同一次请求的所有日志行带上同一个 request_id，便于串联排查
# ---------------------------------------------------------------------------

_request_id_var: ContextVar[str] = ContextVar("request_id", default="-")
_session_id_var: ContextVar[str] = ContextVar("session_id", default="-")


def new_request_id() -> str:
    """生成并设置新的请求 ID（每个进入服务的请求调用一次）"""
    rid = uuid.uuid4().hex[:12]
    _request_id_var.set(rid)
    return rid


def set_request_id(request_id: str) -> None:
    """沿用已存在的请求 ID（如从请求头透传）"""
    _request_id_var.set(request_id or "-")


def get_request_id() -> str:
    """读取当前请求 ID"""
    return _request_id_var.get()


def set_session_id(session_id: Optional[str]) -> None:
    """设置当前请求所属会话 ID"""
    _session_id_var.set(session_id or "-")


def get_session_id() -> str:
    """读取当前会话 ID"""
    return _session_id_var.get()


class _ContextFilter(logging.Filter):
    """把请求上下文注入每条日志记录，供 Formatter 使用"""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = _request_id_var.get()
        record.session_id = _session_id_var.get()
        return True


# ---------------------------------------------------------------------------
# 初始化
# ---------------------------------------------------------------------------

_LOG_FORMAT = (
    "%(asctime)s | %(levelname)-7s | %(request_id)s | %(session_id)s | "
    "%(name)s | %(message)s"
)
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_initialized = False


def setup_logging() -> None:
    """初始化全局日志配置（幂等，重复调用不会重复添加 handler）"""
    global _initialized
    if _initialized:
        return

    settings.ensure_directories()

    level = getattr(logging, str(settings.log_level).upper(), logging.INFO)
    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)
    context_filter = _ContextFilter()

    root = logging.getLogger()
    root.setLevel(level)

    # 先清掉可能已存在的 handler，避免重复输出
    for handler in list(root.handlers):
        root.removeHandler(handler)

    # ---- 文件 handler（按大小滚动，保留历史）----
    file_handler = logging.handlers.RotatingFileHandler(
        filename=str(settings.log_file_path),
        maxBytes=settings.log_max_bytes,
        backupCount=settings.log_backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    file_handler.addFilter(context_filter)
    root.addHandler(file_handler)

    # ---- 控制台 handler ----
    if settings.log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        console_handler.addFilter(context_filter)
        root.addHandler(console_handler)

    # 降噪：第三方库的日志过于啰嗦，只保留警告以上
    for noisy in (
        "urllib3",
        "httpx",
        "httpcore",
        "pymilvus",
        "milvus_lite",
        "modelscope",
        "transformers",
        "sentence_transformers",
        "matplotlib",
        "PIL",
        "filelock",
        "datasets",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _initialized = True
    logging.getLogger(__name__).info(
        "日志系统初始化完成 | 级别=%s | 文件=%s", settings.log_level, settings.log_file_path
    )


def get_logger(name: str) -> logging.Logger:
    """获取 logger。业务代码统一通过本函数获取，保证配置已生效"""
    setup_logging()
    return logging.getLogger(name)


# ---------------------------------------------------------------------------
# 检索链路专用日志辅助
# ---------------------------------------------------------------------------

def log_retrieval_trace(
    logger: logging.Logger,
    *,
    stage: str,
    question: str,
    hits: Optional[list] = None,
    extra: Optional[dict] = None,
) -> None:
    """
    统一记录检索中间过程（Trace）。

    Trace 仅写入服务端日志文件，不对外暴露。各版本 pipeline 复用本函数，
    保证 V1/V2/V3 的日志结构一致，便于横向对比排查。

    参数：
        stage    : 当前环节，如 "dense_retrieve" / "hybrid_rrf" / "rerank" / "rewrite"
        question : 用户原始问题（V3 中同时记录改写后的问题）
        hits     : 命中结果列表，元素为 dict，至少含 chunk_id、score、page_no
        extra    : 该环节特有的附加信息（如耗时、参数）
    """
    parts = [f"[TRACE][{stage}]", f"question={question!r}"]

    if hits:
        # 只打印前若干条，避免日志膨胀
        preview = hits[:10]
        hit_desc = "; ".join(
            "{cid}(score={score:.4f}, page={page})".format(
                cid=str(h.get("chunk_id", "?"))[:12],
                score=float(h.get("score", 0.0)),
                page=h.get("page_no", "?"),
            )
            for h in preview
        )
        parts.append(f"hits={len(hits)}")
        parts.append(f"top{len(preview)}={hit_desc}")

    if extra:
        for key, value in extra.items():
            parts.append(f"{key}={value}")

    logger.info(" | ".join(parts))
