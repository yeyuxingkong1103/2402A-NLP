"""data/index_manifest.json 的读-按 doc_id 合并-写（docs/04 §9.3）。

这份文件是**索引的身份证**，也是后端启动校验的对象。因此它 MUST 与库互为镜像：
`documents[]` 里每个 doc_id 在库里有行，库里的每个 doc_id 也在 `documents[]` 里。

`documents` **以库的实际内容为准重建**，不是把上次的清单直接覆盖 —— 否则
「库里有、清单里没有」的文档会被永久藏起来（FR-022a）。
"""

from __future__ import annotations

import json
import os
from typing import Any

from . import COLLECTION_NAME, DIM, EXIT_VALIDATION, INDEX_TYPE, METRIC_TYPE, IngestError
from . import store


def read_manifest(path: str) -> dict[str, Any] | None:
    """读已有清单。不存在 → None；损坏 → 报错（不静默忽略，否则门禁的差异比对会失去依据）。"""
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise IngestError(
            EXIT_VALIDATION,
            "索引清单损坏：%s\n  %s\n"
            "（不静默忽略：它是版本门禁的差异比对依据。请修复或删除后重跑）" % (path, exc),
        ) from exc
    if not isinstance(data, dict):
        raise IngestError(EXIT_VALIDATION, "索引清单的顶层不是对象：%s" % path)
    return data


def build_manifest(
    client,
    existing: dict[str, Any] | None,
    processed_doc_ids: list[str],
    config: dict[str, Any],
    config_hash: str,
    app_config: dict[str, Any],
    built_at: str,
) -> dict[str, Any]:
    """按 doc_id 合并 + 以库为准核实。计数为 0 的条目被移除。"""
    doc_ids = set(processed_doc_ids)
    if existing:
        for doc in existing.get("documents") or []:
            if isinstance(doc, dict) and doc.get("doc_id"):
                doc_ids.add(str(doc["doc_id"]))

    documents: list[dict[str, Any]] = []
    for doc_id in sorted(doc_ids):
        summary = store.doc_summary(client, doc_id)
        if summary is None:
            continue  # 库为准：已不在库里的条目移除
        documents.append(summary)

    return {
        "built_at": built_at,
        "collection": COLLECTION_NAME,
        "metric_type": METRIC_TYPE,
        "index_type": INDEX_TYPE,
        "dim": DIM,
        "total_chunks": sum(int(d["chunk_count"]) for d in documents),
        "documents": documents,
        "pipeline_config_hash": config_hash,
        # 记的是**输入清单本身**而不只是哈希 —— 门禁失败时要靠它逐项报出哪个参数变了
        "pipeline_config": config,
        # 查询期参数，**不参与** hash（docs/04 §10.2 末段）。改它们无需重建索引。
        "app_config": app_config,
    }


def write_manifest(path: str, data: dict[str, Any]) -> None:
    """先写临时文件再原子替换 —— 中途失败不会留下半份清单。"""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
        os.replace(tmp, path)
    except OSError as exc:
        raise IngestError(EXIT_VALIDATION, "写入索引清单失败：%s\n  %s" % (path, exc)) from exc


def page_counts_of(existing: dict[str, Any] | None) -> dict[str, int]:
    """从已有清单取 page_count，供报告对照（不作为写入来源）。"""
    out: dict[str, int] = {}
    if existing:
        for doc in existing.get("documents") or []:
            if isinstance(doc, dict) and doc.get("doc_id"):
                out[str(doc["doc_id"])] = int(doc.get("page_count") or 0)
    return out


__all__ = ["build_manifest", "page_counts_of", "read_manifest", "write_manifest"]
