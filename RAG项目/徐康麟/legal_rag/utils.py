# -*- coding: utf-8 -*-
"""通用小工具：余弦相似度、归一化、内容寻址 ID、文件指纹、时间戳、摘要。

**为什么单独一个模块**：这些函数被 store / retrieve / ingest / memory 四处共用，
放在这里是为了"同一件事只有一份实现"——尤其 :func:`md5_of_file`（入库指纹）
与 :func:`stable_id`（幂等主键）：它们一旦有两份实现，就会出现
"索引侧算出的 ID ≠ 查询侧算出的 ID"这种极难排查的错位。
"""
from __future__ import annotations

import hashlib
import math
import time


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """余弦相似度；任一为零向量（或长度不匹配的部分为零）时返回 0.0。

    ⚠️ 维度不一致时按**较短的**长度计算（不报错）—— 这是历史行为，
    调用方（内存库/兜底重排）依赖它"永远给一个数"。真机上维度不匹配
    由 store 层在建库时就拦下（见 `store/base.py::_resolve_milvus_dim`）。
    """
    if not a or not b:
        return 0.0
    size = min(len(a), len(b))
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for i in range(size):
        x = a[i]
        y = b[i]
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


def l2_normalize(vector: list[float]) -> list[float]:
    """L2 归一化；零向量原样返回（不制造 NaN）。"""
    norm = math.sqrt(sum(v * v for v in vector))
    if norm <= 0.0:
        return list(vector)
    return [v / norm for v in vector]


def stable_id(*parts: str) -> str:
    """由内容派生稳定 ID，保证同一内容重复入库得到同一主键。

    分隔符用 ``U+241F``（符号形式的单元分隔符）而不是 ``:`` / ``|`` ——
    那种常见分隔符会出现在文本里，导致 ``("a:b", "c")`` 与 ``("a", "b:c")``
    撞成同一个 ID（内容寻址的经典坑）。
    """
    raw = "\u241f".join(str(p) for p in parts)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def md5_of_text(text: str) -> str:
    """文本指纹（UTF-8）。用于"同一段内容是否变过"的判断，不做安全用途。"""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def md5_of_file(path) -> str:
    """文件指纹：**分块读**（1 MB），几十 MB 的 PDF 也不会把内存吃穿。

    入库清单（`index/ingest_manifest.json`）就是靠它与文件比对来决定
    "这篇要不要重新入库"（见 `legal_rag/ingest/pipeline.py`）。
    """
    digest = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def now_ts() -> float:
    """Unix 时间戳（秒，浮点）。单独一个函数是为了测试里能整体打桩。"""
    return time.time()


def make_summary(text: str, limit: int = 120) -> str:
    """先用首段截断做摘要；后续可换成 LLM 总结。"""
    flat = " ".join((text or "").split())
    return flat[:limit]
