"""
quality.py — 分块文本质量过滤

PDF 解析出来的内容里混着页眉页脚、乱码符号、纯分隔线等噪声，
入库前先剔除，否则它们会占掉向量空间并污染检索结果。
"""

from __future__ import annotations

import re

import config

# 允许出现的标点（含中英文），其余字符计入乱码比例
_ALLOWED_PUNCT = set("，。、；：？！“”‘’（）《》〈〉【】〔〕「」·—…～-.,;:?!\"'()[]{}<>/%+*=&#@$~^|`\\_")


def garbage_ratio(text: str) -> float:
    """非常见字符（乱码、控制符、生僻符号）的占比。"""
    if not text:
        return 1.0
    allowed = sum(
        1
        for ch in text
        if ch.isalnum() or ch.isspace() or ch in _ALLOWED_PUNCT or "一" <= ch <= "鿿"
    )
    return 1.0 - allowed / len(text)


def is_meaningless(text: str) -> bool:
    """判断是否为无意义内容：纯符号、重复字符、去掉标点后过短。"""
    stripped = re.sub(r"[\W_]+", "", text, flags=re.UNICODE)
    if len(stripped) < 5:
        return True

    # 单个字符占比过高，例如 "---------" 或 "。。。。。。"
    if stripped:
        most_common = max(stripped.count(ch) for ch in set(stripped))
        if most_common / len(stripped) > 0.8:
            return True

    return False


def filter_chunks(chunks: list[dict]) -> list[dict]:
    """删除过短、乱码、无意义的分块。"""
    kept: list[dict] = []
    for chunk in chunks:
        text = (chunk.get("text") or "").strip()
        if len(text) < config.MIN_CHUNK_LENGTH:
            continue
        if garbage_ratio(text) > 0.3:
            continue
        if is_meaningless(text):
            continue
        kept.append(chunk)
    return kept


if __name__ == "__main__":
    samples = [
        {"text": "劳动合同解除时，用人单位应当向劳动者支付经济补偿。"},
        {"text": "----------"},
        {"text": "。。。"},
        {"text": "　"},
        {"text": "\x00\x01\x02 乱码 ���"},
    ]
    kept = filter_chunks(samples)
    assert len(kept) == 1, f"过滤结果不对：{kept}"
    print("quality 自检通过。")
