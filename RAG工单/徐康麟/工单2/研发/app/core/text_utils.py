"""通用工具函数：文本清洗、分词、ID 生成、数值格式化。

工单：人工智能NLP-RAG-基于PDF文档的问答系统优化
阶段：研发 / 通用工具（被分块、BM25、检索、生成共同复用）

集中放置，避免各模块重复实现；所有函数均为纯函数，便于测试。
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Iterable

# 中文标点与全角字符处理
PUNCT_PATTERN = re.compile(r"[^\w\u4e00-\u9fff]+")
CJK_PATTERN = re.compile(r"[\u4e00-\u9fff]")
DIGIT_PATTERN = re.compile(r"\d")
NUMBER_PATTERN = re.compile(r"\d[\d,，]*\.?\d*")

# 中文停用词（精简版，够用即可；BM25 依赖它压制高频虚词）
STOPWORDS: frozenset[str] = frozenset(
    """
    的 了 和 与 及 或 在 是 为 对 从 到 有 也 就 都 而 并 但 被 把 让 使 于 中 上 下 个 我 你 他
    她 它 们 这 那 这些 那些 什么 怎么 如何 为什么 哪些 哪个 多少 几 请 问 一下 以及 通过 根据
    因此 所以 但是 然而 并且 或者 如果 因为 由于 其中 分别 主要 相关 进行 能够 可以 需要 表示
    一 二 三 四 五 六 七 八 九 十 年 月 日 元 万 亿 % ％
    """.split()
)


def normalize_text(text: str) -> str:
    """归一化空白与全角字符，便于检索对齐。"""
    if not text:
        return ""
    # NFKC 会把全角数字/字母/标点转半角，但会破坏部分中文标点，故仅做空白规整
    text = text.replace("\u3000", " ").replace("\xa0", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.strip() for line in text.split("\n")]
    # 折叠连续空行
    out: list[str] = []
    blank = 0
    for line in lines:
        if not line:
            blank += 1
            if blank > 1:
                continue
        else:
            blank = 0
        out.append(line)
    return "\n".join(out).strip()


def remove_punctuation(text: str) -> str:
    """去掉标点，保留中文、字母、数字。"""
    return PUNCT_PATTERN.sub(" ", text)


def has_cjk(text: str) -> bool:
    """是否包含中日韩文字。"""
    return bool(CJK_PATTERN.search(text))


def tokenize(text: str, use_jieba: bool = True) -> list[str]:
    """分词：优先 jieba，缺失时降级为字符二元组 + 英文单词。

    降级策略对中文检索依然有效（bigram 召回接近分词），保证无 jieba 也能跑。
    """
    if not text:
        return []
    cleaned = remove_punctuation(text).strip()
    if not cleaned:
        return []

    if use_jieba and has_cjk(cleaned):
        try:
            import jieba  # type: ignore

            tokens = [token.strip() for token in jieba.lcut(cleaned)]
            return [t for t in tokens if t and t not in STOPWORDS]
        except Exception:
            pass  # 降级到下面的通用实现

    tokens: list[str] = []
    for word in cleaned.split():
        if not word:
            continue
        if has_cjk(word):
            # 中文：单字 + 相邻二元组，兼顾单字查询与短语匹配
            chars = [ch for ch in word if ch.strip()]
            tokens.extend(chars)
            tokens.extend(chars[i] + chars[i + 1] for i in range(len(chars) - 1))
        else:
            tokens.append(word.lower())
    return [t for t in tokens if t not in STOPWORDS]


def stable_id(prefix: str, *parts: object, length: int = 10) -> str:
    """根据内容生成稳定短 ID（同输入必然同输出，利于索引幂等）。"""
    raw = "|".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha1(raw).hexdigest()[:length]
    return f"{prefix}{digest}"


def format_amount(value: float | int) -> str:
    """金额格式化：1234.5 -> '1,234.50'。"""
    try:
        return f"{float(value):,.2f}"
    except (TypeError, ValueError):
        return str(value)


def extract_numbers(text: str) -> list[str]:
    """抽取文本中的数字串（含千分位），用于答案校验。"""
    return [match.group(0).replace("，", ",") for match in NUMBER_PATTERN.finditer(text or "")]


def truncate(text: str, limit: int = 200, suffix: str = "…") -> str:
    """截断文本，保留语义起点。"""
    if text is None:
        return ""
    text = text.strip()
    return text if len(text) <= limit else text[: limit - len(suffix)] + suffix


def dedupe_keep_order(items: Iterable[str]) -> list[str]:
    """去重且保持首次出现顺序。"""
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def looks_like_heading(line: str) -> bool:
    """判断一行是否为章节标题（招股书结构：第一章/一、（一）/1.1 等）。"""
    stripped = line.strip()
    if not stripped or len(stripped) > 60:
        return False
    patterns = [
        r"^第[一二三四五六七八九十百]+[章节篇]",
        r"^[一二三四五六七八九十]+、",
        r"^（[一二三四五六七八九十]+）",
        r"^\([一二三四五六七八九十]+\)",
        r"^\d+(\.\d+)*[\s、.]",
        r"^[（(]\d+[)）]",
    ]
    return any(re.match(pattern, stripped) for pattern in patterns)


def safe_filename(name: str, limit: int = 80) -> str:
    """把任意标题转成安全的文件名片段。"""
    cleaned = re.sub(r'[\\/:*?"<>|\s]+', "_", name or "").strip("_")
    return cleaned[:limit] or "untitled"


def display_width(text: str) -> int:
    """计算显示宽度（中文算 2）。"""
    return sum(2 if unicodedata.east_asian_width(ch) in {"W", "F"} else 1 for ch in text)
