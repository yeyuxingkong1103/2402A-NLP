# -*- coding: utf-8 -*-
"""功能4：重复检测。MD5 精确去重 + SimHash 近似版本冲突（待确认）。"""
from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

_CHUNK = 1 << 20


def md5_of(path: Path) -> str:
    """分块计算文件 MD5。"""
    h = hashlib.md5()
    with path.open("rb") as f:
        while True:
            block = f.read(_CHUNK)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _feature_hash(token: str, bits: int) -> int:
    return int.from_bytes(hashlib.md5(token.encode("utf-8")).digest()[: bits // 8], "little")


def simhash(text: str, config: dict) -> int:
    """64 位 SimHash。归一化去空白后做 n-gram 特征指纹，numpy 向量化累加。

    特征指纹用 Python 内置 hash（同进程内稳定，满足两两比较；不跨进程持久使用）。
    """
    bits = config["simhash_bits"]
    n = config["ngram"]
    step = config.get("ngram_step", 1)
    normalized = re.sub(r"\s+", "", text)[: config["max_chars_per_doc"]]
    if len(normalized) < n:
        normalized = normalized or " "

    tokens = [normalized[i : i + n] for i in range(0, len(normalized) - n + 1, step)]
    if not tokens:
        return 0

    import numpy as np

    # 确定性 64 位多项式指纹（不使用随机盐的内置 hash，保证跨进程可复现）
    encoded = [t.encode("utf-8") for t in tokens]
    width = max(len(e) for e in encoded)
    mat = np.zeros((len(encoded), width), dtype=np.uint8)
    for i, e in enumerate(encoded):
        mat[i, : len(e)] = list(e)
    # 基数必须 > 255（避免字节数位进位碰撞），且 base^(width-1)*255 < 2^64
    base = np.uint64(1009)
    weights = np.empty(width, dtype=np.uint64)
    weights[0] = 1
    for k in range(1, width):
        with np.errstate(over="ignore"):
            weights[k] = weights[k - 1] * base  # uint64 自动模 2^64，宽度任意安全
    hashes = (mat.astype(np.uint64) * weights).sum(axis=1)  # uint64 自动模 2^64

    fps = hashes.view(np.uint8).reshape(-1, 8)
    # 重复指纹只计一次（等价于 TF 加权但更省内存）；特征已采用长 n-gram，区分度高
    uniq = np.unique(hashes)
    u_fps = uniq.view(np.uint8).reshape(-1, 8)
    bit_grid = np.unpackbits(u_fps, axis=1, bitorder="little")
    sums = bit_grid.astype(np.int64).sum(axis=0) * 2 - bit_grid.shape[0]

    out = np.packbits(sums > 0, bitorder="little").tobytes()
    return int.from_bytes(out[: bits // 8], "little")


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def _similar_snippet(text_a: str, text_b: str, ngram: int = 20) -> str:
    """提取一段可用于人工判断的相似片段：在 a 中找与 b 共享的最长公共窗口。"""
    a = re.sub(r"\s+", "", text_a)[:5000]
    b_set = {text_b[i : i + ngram] for i in range(min(len(text_b) - ngram + 1, 5000))}
    for i in range(0, max(len(a) - ngram + 1, 0), ngram):
        window = a[i : i + ngram]
        if window in b_set:
            return window
    return a[:ngram]


def md5_groups(md5_map: Dict[str, str]) -> List[dict]:
    """md5_map: {file_path: md5}，返回重复分组（size>=2）。"""
    groups = defaultdict(list)
    for fpath, digest in md5_map.items():
        groups[digest].append(fpath)
    result = []
    for digest, files in groups.items():
        if len(files) > 1:
            result.append({"md5": digest, "count": len(files), "files": sorted(files)})
    return sorted(result, key=lambda g: -g["count"])


def simhash_pairs(records: List[dict], config: dict) -> List[dict]:
    """records: [{path, char_count, simhash, text(可选)}]，输出待确认版本冲突列表。

    长度粗筛 + 汉明距离阈值。text 用于生成相似片段（缺失则省略）。
    """
    max_dist = config["hamming_distance"]
    len_ratio = config["length_filter_ratio"]
    text_map = {r["path"]: r.get("text", "") for r in records}
    pairs = []
    n = len(records)
    for i in range(n):
        ri = records[i]
        ci = ri["char_count"] or 1
        for j in range(i + 1, n):
            rj = records[j]
            cj = rj["char_count"] or 1
            if abs(ci - cj) / max(ci, cj) > len_ratio:
                continue
            dist = hamming(ri["simhash"], rj["simhash"])
            if dist <= max_dist:
                snippet = _similar_snippet(text_map.get(ri["path"], ""), text_map.get(rj["path"], ""))
                pairs.append({
                    "file_a": ri["path"],
                    "file_b": rj["path"],
                    "hamming_distance": dist,
                    "similar_snippet": snippet,
                    "status": "待确认",
                })
    return sorted(pairs, key=lambda p: p["hamming_distance"])
