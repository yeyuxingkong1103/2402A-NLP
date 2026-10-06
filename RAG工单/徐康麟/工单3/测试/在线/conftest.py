# -*- coding: utf-8 -*-
"""在线级 conftest（依赖 Ollama / HTTP 服务）。

工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化

本层只负责两件事：
    1. **门控**：Ollama 不可用 → 判失败（除非显式 ``RAG_TEST_ALLOW_ONLINE_SKIP=1``）；
    2. **初始化结构化日志**（app.log / error.log / rag_trace.jsonl 落盘）。
公共 fixture（``engine`` / ``retriever`` / ``golden`` / ``page_lookup`` …）在 ``测试/conftest.py``。
"""

from __future__ import annotations

from typing import Any

import pytest

from common import online_gate


@pytest.fixture(scope="session", autouse=True)
def online_gate_ready(ollama_status: dict[str, Any]) -> None:
    """在线级门控：服务不可用即失败（禁止静默跳过）。"""
    online_gate.enforce(ollama_status)


@pytest.fixture(scope="session", autouse=True)
def logging_ready(app_config: Any) -> Any:
    """初始化产品结构化日志（三文件落盘；供日志断言与 ``simulate_user`` 共用）。"""
    from app.core.logging_conf import setup_logging  # noqa: PLC0415

    setup_logging(app_config, force=True)
    return app_config


@pytest.fixture(scope="session")
def chunk_lookup() -> dict[str, Any]:
    """``chunk_id → chunk 记录`` 回查表（``研发/data/processed/chunks.jsonl``，只读）。

    放在在线级 conftest 是为了让 qa14 与英文覆盖两个用例模块共用（引用核验都要用它）。
    """
    from common import artifacts  # noqa: PLC0415

    return {str(row["chunk_id"]): row for row in artifacts.load_chunks()}
