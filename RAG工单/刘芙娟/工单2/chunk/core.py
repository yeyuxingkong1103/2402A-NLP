"""共用常量、错误类型与长度口径。"""

from __future__ import annotations

import os
import re

from . import RULE_VERSION

# D1 目标区间（非空白字符数）。下限由 500 降为 300：
# 实测 80% 的 section 不到 500 字，而 D1 又只允许同父章节合并——两者叠加
# 会让 62% 的 chunk 落在下限之外。300 是"既不碎、也不过长"的可行解。
LOW, HIGH = 300, 700
# D3 第 1 类：不足下限的这个比例就触发裁决
MIN_RATIO = 0.6
# D3 第 2 类：判定"句子没说完"的最小长度
UNFINISHED_MIN_LEN = 60

EXIT_OK, EXIT_FAIL, EXIT_BAD_INPUT = 0, 1, 2
EXIT_PENDING, EXIT_NO_PAGE = 3, 4

# 自成一块、**不参与合并**的块类型。
#   表格 —— docs/04 §7：整表成块，切了表头后半张表就失去列含义
#   页脚注释 / 图片 / 图表 —— 内容自洽，混进正文会污染引用
#    （实测教训：中图分类号/DOI 那条页脚混进中文引言块，还抢到了块首位置）
STANDALONE_TYPES = frozenset({"table", "page_footnote", "image", "chart", "aside_text"})

TERMINAL = "。！？；：…!?;:"
SENT_BOUNDARY = re.compile(r"(?<=[%s])" % re.escape(TERMINAL))
ENDS_CJK = re.compile(r"[㐀-䶿一-鿿]$")


class ChunkError(Exception):
    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code, self.message = code, message


def project_root() -> str:
    """backend/chunk/core.py -> backend/chunk -> backend -> 仓库根。"""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(os.path.dirname(here))


def char_len(text: str) -> int:
    """非空白字符数——与 S3 清洗的 char_len 口径保持一致。"""
    return sum(1 for ch in text if not ch.isspace())


def size_of(blocks) -> int:
    return sum(char_len(b["text"]) for b in blocks)


__all__ = [
    "LOW", "HIGH", "MIN_RATIO", "UNFINISHED_MIN_LEN", "RULE_VERSION",
    "EXIT_OK", "EXIT_FAIL", "EXIT_BAD_INPUT", "EXIT_PENDING", "EXIT_NO_PAGE",
    "STANDALONE_TYPES", "TERMINAL", "SENT_BOUNDARY", "ENDS_CJK",
    "ChunkError", "project_root", "char_len", "size_of",
]
