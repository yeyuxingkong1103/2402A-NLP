"""读取清洗产物，并按 section 切成待分块单元。"""

from __future__ import annotations

import hashlib
import json
import os

from .core import EXIT_BAD_INPUT, STANDALONE_TYPES, ChunkError, project_root


def load_blocks(path: str) -> list[dict]:
    if not os.path.isfile(path):
        raise ChunkError(EXIT_BAD_INPUT,
                         "清洗产物不存在：%s\n（请先运行 S3：clean_parsed.py）" % path)
    try:
        with open(path, encoding="utf-8") as fh:
            rows = [json.loads(line) for line in fh if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        raise ChunkError(EXIT_BAD_INPUT, "读取失败：%s\n  %s" % (path, exc)) from exc
    if not rows:
        raise ChunkError(EXIT_BAD_INPUT, "清洗产物为空：%s" % path)
    return rows


def resolve_source_hash(doc_id: str) -> str:
    """取完整 64 位哈希用于幂等入库；doc_id 只是它的前 12 位。

    从解析产物里的 _origin.pdf 重算。找不到就退化为 doc_id 并如实反映在报告里。
    """
    parsed = os.path.join(project_root(), "data", "parsed", doc_id)
    for dirpath, _dirs, files in os.walk(parsed):
        for name in files:
            if name.endswith("_origin.pdf"):
                digest = hashlib.sha256()
                with open(os.path.join(dirpath, name), "rb") as fh:
                    for chunk in iter(lambda: fh.read(1 << 20), b""):
                        digest.update(chunk)
                return digest.hexdigest()
    return doc_id


def embed_text(block: dict) -> str:
    """算相似度用的文本：标题链 + 正文，与 chunk 的 text_for_embedding 同口径。"""
    head = " > ".join(block["heading_path"])
    return "%s\n\n%s" % (head, block["text"]) if head else block["text"]


def build_units(blocks: list[dict]) -> list[dict]:
    """按 section 切单元。

    - 标题块是**硬边界**，只贡献 heading_path，不产出自己的文本
    - STANDALONE_TYPES（表格/页脚注释/图片）**独占一段**，不与正文混
    """
    units: list[dict] = []
    cur: dict | None = None
    for block in blocks:
        if block["is_heading"]:
            cur = None
            continue
        key = tuple(block["heading_path"])
        if cur is None or cur["key"] != key:
            cur = {"key": key, "segments": [], "pending": []}
            units.append(cur)
        if block["block_type"] in STANDALONE_TYPES:
            if cur["pending"]:
                cur["segments"].append(cur["pending"])
                cur["pending"] = []
            cur["segments"].append([block])
        else:
            cur["pending"].append(block)

    for unit in units:
        if unit["pending"]:
            unit["segments"].append(unit["pending"])
        unit.pop("pending")
    return [u for u in units if u["segments"]]
