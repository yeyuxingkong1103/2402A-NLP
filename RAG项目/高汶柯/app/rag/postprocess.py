"""后处理：正则清洗 + 引用校验。"""
from __future__ import annotations

import re

_MULTI_BLANK = re.compile(r"\n{3,}")
_MULTI_SPACE = re.compile(r"[ \t]{2,}")
_HEADING = re.compile(r"^#{1,6}\s*", flags=re.M)
_CITATION = re.compile(r"《[^》]{2,30}》第[零一二三四五六七八九十百千万两\d]+条")


def clean(text: str) -> str:
    """去除多余空行、空白与 Markdown 标题符号。"""
    text = _MULTI_BLANK.sub("\n\n", text)
    text = _MULTI_SPACE.sub(" ", text)
    text = _HEADING.sub("", text)
    return text.strip()


def extract_citations(text: str) -> list[str]:
    """提取形如《法律名称》第X条 的引用。"""
    return _CITATION.findall(text)


def postprocess(text: str, context: str = "", verify: bool = True) -> str:
    """清洗回答，并对未在资料中出现的法条引用给出提示。"""
    text = clean(text)
    if not verify or not context:
        return text
    unverified = [c for c in extract_citations(text) if c not in context]
    if unverified:
        unique = list(dict.fromkeys(unverified))
        text += "\n\n> 提示：以下引用未在知识库原文中检索到，请进一步核实：" + "、".join(unique)
    return text
