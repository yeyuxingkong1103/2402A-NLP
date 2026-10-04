"""文本清洗：去水印、归一化、低质量过滤、去重。"""
from __future__ import annotations

import hashlib
import re

_CJK = r"\u4e00-\u9fa5"
_WS_BETWEEN_CJK = re.compile(rf"(?<=[{_CJK}])\s+(?=[{_CJK}])")
_MULTI_BLANK = re.compile(r"\n{3,}")
_MULTI_SPACE = re.compile(r"[ \t]{2,}")

# 常见水印 / 噪声行
WATERMARK_PATTERNS = [
    re.compile(r"^.{0,6}第?\s*\d+\s*页.{0,6}$"),
    re.compile(r"^[-—_\s]*\d{1,3}[-—_\s]*$"),
    re.compile(r"https?://\S+"),
    re.compile(r"www\.\S+"),
    re.compile(r"仅供.{0,10}(参考|学习|交流)"),
    re.compile(r"扫描全能王|道客巴巴|百度文库|CSDN"),
]


def remove_watermarks(text: str) -> str:
    """逐行剔除水印与页眉页脚噪声。"""
    lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            lines.append("")
            continue
        if len(stripped) <= 30 and any(p.search(stripped) for p in WATERMARK_PATTERNS):
            continue
        lines.append(line)
    return "\n".join(lines)


def normalize(text: str) -> str:
    """归一化：去水印、去除汉字间多余空格、压缩空白。"""
    text = remove_watermarks(text)
    text = _WS_BETWEEN_CJK.sub("", text)
    text = _MULTI_SPACE.sub(" ", text)
    text = _MULTI_BLANK.sub("\n\n", text)
    return text.strip()


def is_low_quality(text: str, min_len: int = 15) -> bool:
    """过滤过短或噪声占比过高的片段。"""
    t = text.strip()
    if len(t) < min_len:
        return True
    noise = len(re.findall(r"[^\w\u4e00-\u9fa5]", t))
    return noise / max(len(t), 1) > 0.6


def _norm_key(text: str) -> str:
    return hashlib.md5(re.sub(r"\s+", "", text).encode("utf-8")).hexdigest()


def dedup(chunks: list[dict]) -> list[dict]:
    """基于归一化内容哈希去重。"""
    seen: set[str] = set()
    out: list[dict] = []
    for c in chunks:
        key = _norm_key(c["text"])
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def clean_chunks(chunks: list[dict]) -> list[dict]:
    """过滤低质量 + 去重。"""
    kept = [c for c in chunks if not is_low_quality(c["text"])]
    return dedup(kept)
