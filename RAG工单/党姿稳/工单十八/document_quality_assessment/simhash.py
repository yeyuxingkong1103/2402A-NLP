# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-18-实现文档质量评估Skill并集成至智能体工作流
# 模块职责：SimHash 近似重复指纹（中文按字符二元组切词）与汉明距离计算；
#           近似重复必须人工确认，因此这里只产出候选与相似度证据，不做自动删除。

import re                                                          # 文本清洗
import zlib                                                        # 稳定哈希（跨进程可复现）

_CJK = re.compile(r"[一-龥]")                              # 中文字符
_WS = re.compile(r"\s+")                                           # 空白
# 中文文档用字符二元组比按词切分更稳，且不需要额外词典；英文数字按词切分
_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")


def _tokens(text, limit=20000):
    """把文本切成用于 SimHash 的特征集合：中文二元组 + 英文数字词。"""
    clean = _WS.sub("", text)[:limit]                              # 去掉空白并限长，控制耗时
    grams = []                                                     # 特征列表
    for i in range(len(clean) - 1):                                # 字符二元组
        pair = clean[i:i + 2]
        if _CJK.search(pair):                                      # 至少含一个汉字才算中文特征
            grams.append(pair)
    grams.extend(t.lower() for t in _TOKEN_RE.findall(text[:limit]))  # 英文数字词
    return grams


def _hash64(token):
    """把特征映射为 64 位无符号整数（用 crc32 保证跨进程稳定）。"""
    return zlib.crc32(token.encode("utf-8", "ignore")) & 0xFFFFFFFF


def _hash_pair(token):
    """返回该特征对应的 64 位哈希（高低 32 位拼接，降低碰撞）。"""
    lo = _hash64(token)                                            # 低 32 位
    hi = _hash64(token + "\x00salt")                               # 高 32 位（加盐区分同名特征）
    return (hi << 32) | lo


def fingerprint(text, bits=64):
    """计算 SimHash 指纹，返回整数；位宽只支持 64。"""
    toks = _tokens(text)                                           # 特征
    if not toks:                                                   # 空文本无指纹
        return None
    weights = [0] * 64                                             # 每位的加权和
    for tok in toks:                                               # 逐特征投票
        h = _hash_pair(tok)                                        # 特征哈希
        for b in range(64):                                        # 逐位累加 ±1
            weights[b] += 1 if (h >> b) & 1 else -1
    value = 0                                                      # 指纹结果
    for b in range(64):                                            # 正权重位取 1
        if weights[b] > 0:
            value |= (1 << b)
    return value


def hamming(a, b):
    """两个指纹的汉明距离；任一为 None 时返回极大值。"""
    if a is None or b is None:
        return 999
    return bin((a ^ b) & 0xFFFFFFFFFFFFFFFF).count("1")


def bands(value, count=4, bits=64):
    """把指纹切成 count 段，用于分桶筛候选，避免全量两两比对。"""
    seg = bits // count                                            # 每段位宽
    mask = (1 << seg) - 1                                          # 段掩码
    return [(i, (value >> (i * seg)) & mask) for i in range(count)]


def similarity(a, b):
    """把汉明距离换算成 0~1 的相似度，便于报告中排序展示。"""
    return round(1.0 - hamming(a, b) / 64.0, 4)
