# 工单编号：人工智能NLP-RAG 项目-Embedding 模型微调任务
# -*- coding: utf-8 -*-
"""文本处理与相似度小工具（无第三方依赖，纯标准库 + 正则）。"""
from __future__ import annotations

import re
import unicodedata
from typing import Iterable

_WS = re.compile(r"\s+")
_COMMA = re.compile(r"[,\uFF0C]")
_SENT_SPLIT = re.compile(r"[。；;！？!?\n]+")

# 全角 -> 半角
_FULL2HALF = {i: i - 0xFEE0 for i in range(0xFF01, 0xFF5F)}
_FULL2HALF[0x3000] = 0x20


def normalize_text(s: str) -> str:
    """归一化：全角转半角、去空白、去千分位逗号、统一大小写。"""
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", str(s))
    s = s.translate(_FULL2HALF)
    s = _WS.sub("", s)
    s = _COMMA.sub("", s)
    return s.lower()


def clean_display(s: str, limit: int = 0) -> str:
    s = _WS.sub(" ", str(s or "")).strip()
    if limit and len(s) > limit:
        return s[: limit - 1] + "…"
    return s


def sentences(text: str, min_len: int = 8) -> list[str]:
    out: list[str] = []
    for part in _SENT_SPLIT.split(str(text or "")):
        part = part.strip()
        if len(part) >= min_len:
            out.append(part)
    return out


def char_ngrams(s: str, n: int = 2) -> set[str]:
    s = normalize_text(s)
    if len(s) < n:
        return {s} if s else set()
    return {s[i : i + n] for i in range(len(s) - n + 1)}


def jaccard(a: str, b: str, n: int = 2) -> float:
    """字符 n-gram Jaccard 相似度（用于弱监督「带相似度分数的句子对」）。"""
    sa, sb = char_ngrams(a, n), char_ngrams(b, n)
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return round(inter / union, 6) if union else 0.0


def parse_doc_name(file_name: str) -> dict[str, str]:
    """从工单7 分块文件名解析文档元信息。

    形如 ``2020-02-14__平安银行股份有限公司__000001__平安银行__2019年__年度报告.pdf``
    """
    parts = str(file_name or "").split("__")
    info = {"file_name": file_name, "日期": "", "全称": "", "代码": "", "简称": "", "年份": "", "报告类型": ""}
    if len(parts) >= 6:
        info.update({"日期": parts[0], "全称": parts[1], "代码": parts[2], "简称": parts[3],
                     "年份": parts[4], "报告类型": parts[5].replace(".pdf", "")})
    elif len(parts) == 1:
        info["全称"] = parts[0]
    return info


def extract_numbers(text: str) -> list[str]:
    return re.findall(r"\d[\d,\.]*", str(text or ""))


def keyword_hit_count(text: str, keywords: Iterable[str]) -> tuple[int, list[str]]:
    """统计判分关键词在文本中的命中数（归一化后子串匹配）。"""
    norm = normalize_text(text)
    hits = []
    for kw in keywords or []:
        k = normalize_text(kw)
        if k and k in norm:
            hits.append(kw)
    return len(hits), hits
