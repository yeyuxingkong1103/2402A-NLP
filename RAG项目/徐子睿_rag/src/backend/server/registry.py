# -*- coding: utf-8 -*-
"""server/registry.py —— 文档登记表的读写与体积统计。

在链路中的位置：
    构建完成后把结果写进 data/documents.json；知识库与文档列表接口读它。

登记表的内容：
    每份 PDF 一行：文件名、大小、页数、chunk 数、向量数、构建耗时、构建时间。
    它记录的是"构建当时发生了什么"，与 Milvus 里的实时存量是两回事
    （/api/kb/overview 以 Milvus 为准，本文件的列表以登记表为准）。
"""
from __future__ import annotations

import json
from typing import Any

from .config import DATA_DIR, DOCS_JSON

def load_docs() -> list[dict[str, Any]]:
    """读取文档登记表。

    返回：
        登记信息列表；文件不存在时返回空列表（首次启动的正常情况）。
    """
    if not DOCS_JSON.exists():
        return []
    return json.loads(DOCS_JSON.read_text(encoding="utf-8"))

def save_docs(docs: list[dict[str, Any]]) -> None:
    """写回文档登记表。

    参数：
        docs: 完整列表（调用方负责增删）
    说明：
        ensure_ascii=False 让中文文件名/字段在文件里可读，便于人工排查。
    """
    DOCS_JSON.write_text(json.dumps(docs, ensure_ascii=False, indent=2), encoding="utf-8")

def update_doc(doc: dict[str, Any]) -> None:
    """插入或更新一条文档登记信息。

    参数：
        doc: 含 filename 及构建统计的字典

    先按 filename 过滤再 append，等价于"有则覆盖、无则新增"：
        同名 PDF 重复上传时，登记表里只应留最新一条，否则列表页会出现重复行。
    """
    docs = [item for item in load_docs() if item.get("filename") != doc["filename"]]
    docs.append(doc)
    save_docs(docs)

def data_size() -> int:
    """统计 data 目录占用的总字节数（知识库规模展示用）。"""
    return sum(path.stat().st_size for path in DATA_DIR.rglob("*") if path.is_file())
