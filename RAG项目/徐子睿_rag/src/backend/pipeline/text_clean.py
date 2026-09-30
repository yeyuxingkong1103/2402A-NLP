# -*- coding: utf-8 -*-
"""pipeline/text_clean.py —— 术语归一化与噪声过滤。

在链路中的位置：
    backend/pipeline 构建管线的第二步：把解析出来的原始文本洗成"同一种写法的干净文本"。

清洗的目的不是让文字好看，而是让同一个术语以同一种形式进入检索系统 ——
"SF 6"、"GB / T 44653 - 2024" 这类 PDF 里被拆开的写法如果不清洗，
用户按规范写法搜索时就永远命不中。

四个函数的顺序即处理顺序：
    normalize_text   单行术语归一化与空白规整
    is_noise         判断该行是否该丢弃
    clean_text       逐行清洗整页
    keep_body_pages  裁掉封面/目录/前言，只留正文
"""
from __future__ import annotations

import re

from .config import HEADER_RE, TOC_RE

# ------------------------------ 文本清洗和分块


def normalize_text(text: str) -> str:
    """对单行文本做术语归一化与空白规整。

    参数：
        text: 原始单行文本
    返回：
        归一化后的文本（首尾空白已去除）。

    为什么这一步比它看起来重要：
        "同一个术语必须以同一种形式进入检索系统"。PDF 里同一个意思可能有多种写法，
        如果入库时写法不统一，用户搜 A 写法就永远命不中存成 B 写法的片段。
        这里处理的都是实际踩过的坑：
        - 三种破折号（⁃ — －）统一成半角 -，否则标准号 "GB/T 44653—2024" 搜不到
        - "SF 6"（MinerU 把下标 6 提成独立文本）统一成 "SF6"
        - "GB / T 44653 - 2024" 统一成 "GB/T 44653-2024"
        - 中文之间被 PDF 强行插入的空格去掉（PDF 常把每个字当独立文本块）
    """
    text = text.replace("⁃", "-").replace("—", "-").replace("－", "-")
    # 下面三条按"从特殊到一般"的顺序处理标准号，顺序不能反，否则宽松规则会先吃掉严格规则
    text = re.sub(r"\b(GB|DL)\s*/\s*T\s*(\d+(?:\s*\.\s*\d+)?)\s*-\s*(\d{3})\s*(\d)\b", r"\1/T \2-\3\4", text, flags=re.I)
    text = re.sub(r"\b(GB|DL)\s*/\s*T\s*(\d+(?:\s*\.\s*\d+)?)\s*-\s*(\d{4})\b", r"\1/T \2-\3", text, flags=re.I)
    text = re.sub(r"\b(GB|DL)\s*/\s*T\s*(?=\d)", lambda match: match.group(1).upper() + "/T ", text, flags=re.I)
    text = re.sub(r"\bSF\s+6\b", "SF6", text, flags=re.I)  # SF6 下标被提成普通文本的情况
    text = re.sub(r"(?<=\d)\s*\.\s*(?=\d)", ".", text)      # 数字中间的空格点号，如 "3 . 2" -> "3.2"（章节号）
    text = re.sub(r"[ \t　]+", " ", text)                    # 连续空白（含全角空格）压成一个
    text = re.sub(r"\s+([，。；：、！？）】》])", r"\1", text)  # 去掉标点前的空格
    text = re.sub(r"\s*([/\-])\s*", r"\1", text)             # 斜杠/连字符两侧不留空格
    text = re.sub(r"(?<=[一-鿿])\s+(?=[一-鿿])", "", text)   # 中文之间被切开插入的空格，直接删
    return text.strip()

def is_noise(line: str) -> bool:
    """判断一行是否属于应当丢弃的噪声。

    参数：
        line: 单行文本
    返回：
        True 表示这行应该被丢掉。

    判为噪声的三类：
        1. 空行，或整行只有数字/标点/罗马数字（页码、分隔线）
        2. 匹配页眉页脚正则（罗马数字页码、反复出现的标准号）
        3. 目录行：有点线且去掉点线数字后剩余内容不足 18 字
           —— 长度限制是为了不误杀正文里正常使用的省略号
    """
    line = line.strip()
    return (
        not line
        or bool(re.fullmatch(r"[\d\s.\-—_/·…ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+", line))
        or bool(HEADER_RE.fullmatch(line))
        or (bool(TOC_RE.search(line)) and len(re.sub(r"[·.。…\s\d]", "", line)) < 18)
    )

def clean_text(text: str) -> str:
    """逐行清洗整页文本：先归一化，再滤噪声，最后清理行尾残留的点线页码。

    参数：
        text: 一页的原始文本（多行）
    返回：
        清洗后的多行文本。

    注意这里清了两遍 is_noise：
        第一步之后还会删掉行尾的 "···· 12" 尾巴，删完这一行可能才真正变成噪声，
        所以要再判一次。否则会留下只有几个字加一串点的残行进入向量库。
    """
    lines = []
    for line in text.splitlines():
        line = normalize_text(line)
        if is_noise(line):
            continue
        line = re.sub(r"\s*[·.。…]{2,}\s*\d*\s*$", "", line).strip()  # 去掉行尾 "…… 12" 这类目录残余
        if not is_noise(line):
            lines.append(line)
    return "\n".join(lines)

def keep_body_pages(pages: list[dict]) -> list[dict]:
    """截掉封面、目录、前言，只保留正文。

    参数：
        pages: clean_text 处理后的按页结构
    返回：
        从正文第一页开始的页列表。

    定位方式：
        以 "1 范围" 作为正文起点 —— 这是国标/行业标准文档的固定结构。
        找不到该标题时原样返回全部内容（**宁可多留，不可误删**），
        因为不同文档的正文起始标记并不统一，误删会造成不可逆的信息丢失。
    """
    body, started = [], False
    for page in pages:
        lines = []
        for line in page["text"].splitlines():
            if not started and re.match(r"^\s*1\s+范围\b", line.strip()):
                started = True
            if started:
                lines.append(line)
        if lines:
            body.append({"page": page["page"], "text": "\n".join(lines)})
    return body or pages  # 一页正文都没截到时返回原文，避免整篇文档被清空
