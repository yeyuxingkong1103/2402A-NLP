# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
模块说明：通用工具，包含日志、计时与文本清洗函数。
"""
import logging
import re
import time
from contextlib import contextmanager

import config

_LOGGER_NAME = "pdf_qa"


def get_logger(name: str = _LOGGER_NAME) -> logging.Logger:
    """获取全局日志器（重复调用不会重复添加 handler）。"""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                f"[{config.WORK_ORDER_NO}] %(asctime)s %(levelname)s %(name)s: %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.addHandler(handler)
        logger.setLevel(getattr(logging, config.LOG_LEVEL.upper(), logging.INFO))
        logger.propagate = False
    return logger


logger = get_logger()


@contextmanager
def timer():
    """上下文管理器：统计代码块耗时（秒），结果写入 result 字典。"""
    result = {"elapsed": 0.0}
    start = time.perf_counter()
    try:
        yield result
    finally:
        result["elapsed"] = round(time.perf_counter() - start, 3)


def clean_text(text: str) -> str:
    """清洗文本：合并多余空白、去掉孤立页码与不可见字符。"""
    if not text:
        return ""
    text = text.replace("\u3000", " ").replace("\xa0", " ")
    text = re.sub(r"[\u200b\u200e\u200f\ufeff]", "", text)
    # 连续 3 个以上换行压缩为 2 个
    text = re.sub(r"\n{3,}", "\n\n", text)
    # 行内多余空格压缩
    text = re.sub(r"[ \t]{2,}", " ", text)
    # 去掉仅由“第 x 页”/纯数字组成的行
    lines = [ln for ln in text.split("\n") if not re.fullmatch(r"\s*(第?\s*\d+\s*页?|\d{1,4})\s*", ln)]
    return "\n".join(lines).strip()
