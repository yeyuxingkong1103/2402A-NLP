"""pipeline_config_hash：影响产物的全部参数的哈希（docs/04 §10.2）。

**数值参数从上游代码 import，不从文档抄。**

这条是 D2 裁决的核心。实测教训：`docs/04 §7` 写着「300–500 字」、`specs/003` 的 D1
裁决写着「500–700 字」、而 `backend/chunk/core.py` 实际用的是 `LOW, HIGH = 300, 700`
—— 三个不同的值。若本模块照文档抄，算出来的就是一个**从未真正用过的参数**的哈希：
门禁看起来在工作，实际永远不触发。

从 `chunk.core` import 之后，S6 与 S4 共用同一个常量源，改一处两边同时变。
（实测确认 `chunk/core.py` 只 import os/re，不拉 torch —— 这个 import 是零成本的。）
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from chunk.core import HIGH, LOW, MIN_RATIO  # noqa: F401  —— 单一数据源，勿改为字面量
from clean import CLEAN_RULE_VERSION

# S4 的分块是「在边界处切」，无滑窗重叠；产物里也没有 overlap 字段。
# 写成常量是为了让它进入 hash —— 将来若引入重叠，改这里即可让门禁触发。
CHUNK_OVERLAP = 0

# 查询期参数（top_k / 相似度阈值）**不参与** hash（docs/04 §10.2 末段）。
# 改它们不需要重建索引，把它们写进 hash 是常见的过度设计。


def collect_pipeline_config(chunk_rule_version: str, fingerprint: dict[str, Any]) -> dict[str, Any]:
    """汇总本次的 pipeline_config。返回的 dict 原样写进 manifest，供门禁失败时逐项比对。"""
    return {
        "clean_rule_version": CLEAN_RULE_VERSION,
        "chunk_rule_version": chunk_rule_version,
        # 数值来自 backend/chunk/core.py（实际生效值），不是文档
        "chunk_min_chars": LOW,
        "chunk_max_chars": HIGH,
        "min_ratio": MIN_RATIO,
        "chunk_overlap": CHUNK_OVERLAP,
        # S5 已经把「影响向量数值的全部参数」收进指纹，整份纳入：既不漏也不挑
        "embed": dict(fingerprint),
    }


def compute_hash(config: dict[str, Any]) -> str:
    """确定性哈希：键按字典序、UTF-8、ensure_ascii=False。

    同一组输入恒得同一值，且与本文件随附的键顺序无关 —— 否则改动 dict 字面量的
    书写顺序就会让全库需要重建。
    """
    blob = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def describe_diff(old: dict[str, Any] | None, new: dict[str, Any]) -> list[str]:
    """逐项列出差异参数（US3-2：门禁报错必须点出**哪个**参数变了）。

    old 为 None 表示拿不到旧值（manifest 缺失或不可读）—— 这时如实说明，
    **不假装知道差异在哪**。
    """
    if old is None:
        return ["（旧参数清单不可得：data/index_manifest.json 缺失或不可读，只能比对哈希值本身）"]

    lines: list[str] = []
    for key in sorted(set(old) | set(new)):
        if key == "embed":
            continue  # 展开到下一层
        before, after = old.get(key, "<缺失>"), new.get(key, "<缺失>")
        if before != after:
            lines.append("  %-20s %s  →  %s" % (key, before, after))

    old_embed = old.get("embed") or {}
    new_embed = new.get("embed") or {}
    for key in sorted(set(old_embed) | set(new_embed)):
        before, after = old_embed.get(key, "<缺失>"), new_embed.get(key, "<缺失>")
        if before != after:
            lines.append("  embed.%-14s %s  →  %s" % (key, before, after))

    return lines or ["（逐项比对未发现差异，但哈希不同 —— 说明两边的参数清单结构不一致，需人工核对）"]


__all__ = ["CHUNK_OVERLAP", "collect_pipeline_config", "compute_hash", "describe_diff"]
