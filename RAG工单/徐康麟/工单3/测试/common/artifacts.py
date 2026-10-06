# -*- coding: utf-8 -*-
"""T8 解析/索引产物读取（只读，统一入口）。

工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化

产物布局（``设计/接口设计.md`` §5.2/§5.4）：
    ``研发/data/processed/{stem}.pages.jsonl``       页面文本
    ``研发/data/processed/{stem}.tables.jsonl``      表格块（含 markdown / degenerate / logical_cols）
    ``研发/data/processed/chunks.jsonl``             全量分块
    ``研发/data/index/<model_slug>/``                vectors.npy / ids.json / bm25.pkl / index_manifest.json

纪律：全部只读；文件缺失时**显式抛错**并给出可执行的重建命令，不静默返回空集合。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from . import paths


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """读 JSONL（坏行抛 ``ValueError``，避免把解析失败当成「数据为空」）。"""
    if not path.exists():
        raise FileNotFoundError(
            f"产物缺失：{paths.display(path)}；重建命令："
            f"pwsh -NoProfile -File run_py.ps1 研发/scripts/parse_corpus.py")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno} JSONL 解析失败：{exc}") from exc
    return rows


def stem_for(corpus: str) -> str:
    """语料别名（``pdf1``/``pdf2``/``1``/``2``）→ 文件名去扩展名的 stem。"""
    path = paths.resolve_corpus("2" if str(corpus).endswith("2") else "1")
    if path is None:
        raise FileNotFoundError(f"语料未自动发现：corpus={corpus!r}，目录 {paths.display(paths.RAW_DIR)}")
    return path.stem


def load_pages(corpus: str) -> list[dict[str, Any]]:
    """页面文本产物（``PageText`` 列表，1-based ``page`` 字段）。"""
    return read_jsonl(paths.processed_jsonl(stem_for(corpus), "pages"))


def load_tables(corpus: str) -> list[dict[str, Any]]:
    """表格块产物（``TableBlock`` 列表）。"""
    return read_jsonl(paths.processed_jsonl(stem_for(corpus), "tables"))


def load_text_blocks(corpus: str) -> list[dict[str, Any]]:
    """文本块产物（``TextBlock`` 列表）。"""
    return read_jsonl(paths.processed_jsonl(stem_for(corpus), "text_blocks"))


def load_chunks() -> list[dict[str, Any]]:
    """全量分块产物（``Chunk`` 列表，跨语料）。"""
    return read_jsonl(paths.chunks_jsonl())


def tables_on_page(corpus: str, page: int) -> list[dict[str, Any]]:
    """取某物理页上的表格块（按 ``page_start`` 或 ``row_pages`` 命中）。"""
    hit: list[dict[str, Any]] = []
    for block in load_tables(corpus):
        pages = {int(block.get("page_start", 0)), int(block.get("page_end", 0))}
        pages.update(int(p) for p in block.get("row_pages", []) or [])
        pages.update(int(p) for p in block.get("absorbed_pages", []) or [])
        if int(page) in pages:
            hit.append(block)
    return hit


def chunks_for_file(file_name: str) -> list[dict[str, Any]]:
    """某语料的全部 chunk（用于「退化表不进索引」「页数一致」等断言）。"""
    return [c for c in load_chunks() if str(c.get("file_name")) == file_name]


def manifest(model_slug: str = "bge-m3_1024") -> dict[str, Any]:
    """``index_manifest.json``（T4 产出，T8 用于核对页数/块数/模型/维度）。"""
    path = paths.index_model_dir(model_slug) / "index_manifest.json"
    if not path.exists():
        raise FileNotFoundError(
            f"索引 manifest 缺失：{paths.display(path)}；重建命令："
            f"pwsh -NoProfile -File run_py.ps1 研发/scripts/build_index.py")
    return json.loads(path.read_text(encoding="utf-8"))


def iter_all_kinds(corpus: str) -> Iterable[tuple[str, list[dict[str, Any]]]]:
    """产出 ``(产物名, 行列表)``，便于汇总统计。"""
    yield "pages", load_pages(corpus)
    yield "tables", load_tables(corpus)
    yield "text_blocks", load_text_blocks(corpus)
