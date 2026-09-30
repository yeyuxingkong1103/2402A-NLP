# -*- coding: utf-8 -*-
"""输出后处理：敏感词替换、格式校验、Markdown 清洗、流式输出。

对应设计文档里的「后处理：正则替换敏感词、格式校验、Markdown 清洗、流式输出」。
"""
from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# 保守的示例名单，实际项目应改为从配置/词库加载
DEFAULT_SENSITIVE_WORDS: tuple[str, ...] = (
    "操你", "傻逼", "妈的", "滚蛋",
)

_MULTI_BLANK_RE = re.compile(r"\n{3,}")
_TRAILING_SPACE_RE = re.compile(r"[ \t]+\n")
_UNCLOSED_BOLD_RE = re.compile(r"\*\*(?!.*\*\*)")


def sanitize(text: str, extra_words: list[str] | None = None) -> str:
    """把敏感词替换为等长的 *。"""
    words = list(DEFAULT_SENSITIVE_WORDS) + list(extra_words or [])
    result = text or ""
    for word in words:
        if word:
            result = result.replace(word, "*" * len(word))
    return result


def clean_markdown(text: str) -> str:
    """压掉多余空行与行尾空白，避免流式拼接出现奇怪格式。"""
    result = _TRAILING_SPACE_RE.sub("\n", text or "")
    result = _MULTI_BLANK_RE.sub("\n\n", result)
    return result.strip()


def validate(text: str, min_length: int = 1) -> tuple[bool, str]:
    if not text or not text.strip():
        return False, "回答为空"
    if len(text.strip()) < min_length:
        return False, f"回答过短（{len(text.strip())} < {min_length}）"
    return True, ""


def postprocess(text: str, extra_sensitive: list[str] | None = None,
                do_clean: bool = True) -> str:
    """完整后处理管线。校验失败时也会返回一个可展示的兜底文案。"""
    result = sanitize(text, extra_sensitive)
    if do_clean:
        result = clean_markdown(result)
    ok, reason = validate(result)
    if not ok:
        logger.warning("后处理校验未通过：%s", reason)
        result = "抱歉，本次未能生成有效回答，请换一种问法或补充信息后重试。"
    return result
