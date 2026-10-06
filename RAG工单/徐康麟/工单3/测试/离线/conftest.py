# -*- coding: utf-8 -*-
"""离线级 conftest（不依赖 Ollama / HTTP 服务）。

工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化

离线级只做**确定性**断言：产物结构、表格归一化锚点、分块元数据、BM25 索引、
配置契约、日志结构与静态检查；**不做**端到端问答（那属于在线级）。
"""

from __future__ import annotations

from typing import Any

import pytest

from common import artifacts, paths


@pytest.fixture(scope="session")
def pdf1_stem() -> str:
    """PDF1（兴图新科）的 stem（自动发现，不硬编码文件名）。"""
    return artifacts.stem_for("pdf1")


@pytest.fixture(scope="session")
def pdf2_stem() -> str:
    """PDF2（力源信息）的 stem；语料缺席时显式 skip 并说明原因。"""
    stem = None
    try:
        stem = artifacts.stem_for("pdf2")
    except FileNotFoundError as exc:
        pytest.skip(f"PDF2 语料缺席，跳过 PDF2 相关离线断言：{exc}")
    return stem


@pytest.fixture(scope="session")
def pages_pdf1(pdf1_stem: str) -> list[dict[str, Any]]:
    """PDF1 页面文本产物。"""
    return artifacts.load_pages(pdf1_stem)


@pytest.fixture(scope="session")
def tables_pdf1(pdf1_stem: str) -> list[dict[str, Any]]:
    """PDF1 表格块产物。"""
    return artifacts.load_tables(pdf1_stem)


@pytest.fixture(scope="session")
def tables_pdf2(pdf2_stem: str) -> list[dict[str, Any]]:
    """PDF2 表格块产物。"""
    return artifacts.load_tables(pdf2_stem)


@pytest.fixture(scope="session")
def chunks_all() -> list[dict[str, Any]]:
    """全量分块产物。"""
    return artifacts.load_chunks()


@pytest.fixture(scope="session")
def index_manifest() -> dict[str, Any]:
    """索引 manifest（T4 产出）。"""
    return artifacts.manifest()


@pytest.fixture(scope="session")
def bm25_index() -> Any:
    """BM25 索引对象（本地确定性，无需嵌入服务）。

    注意：``BM25Index.load`` 的入参是**索引目录**（内部会拼 ``bm25.pkl``），
    传文件路径会得到 ``.../bm25.pkl/bm25.pkl`` 这种错误路径。
    """
    paths.ensure_dev_on_path()
    from app.core.bm25_index import BM25Index  # noqa: PLC0415

    index_dir = paths.index_model_dir("bge-m3_1024")
    if not (index_dir / "bm25.pkl").exists():
        pytest.fail(f"BM25 索引缺失：{paths.display(index_dir)}；重建命令："
                    f"pwsh -NoProfile -File run_py.ps1 研发/scripts/build_index.py")
    return BM25Index.load(index_dir)
