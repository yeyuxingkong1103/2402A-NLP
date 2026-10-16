# -*- coding: utf-8 -*-
"""工单8 pytest 公共夹具与工具。

工单编号：人工智能NLP-RAG-基于Graph RAG 实现金融问答

设计纪律：
    * **重资源只加载一次**（进程级 fixture）：8847 块索引 + 图谱加载约 4 s，
      每题都重载会让测试跑到分钟级；
    * 图谱/索引缺失时**跳过并给出明确原因**（`pytest.skip`），
      不把「环境没准备好」误报成「功能坏了」；
    * 不联网、不起服务、不写工单7 目录。
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_REPO_RESEARCH = _ROOT / "研发"
if str(_REPO_RESEARCH) not in sys.path:
    sys.path.insert(0, str(_REPO_RESEARCH))

#: 测试期间禁止写 .pyc（工单1 venv 是只读共享资源）
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

GRAPH_DIR = _ROOT / "研发" / "data" / "graph"
TEST_DATA_DIR = _ROOT / "测试" / "测试数据"
SHOT_DIR = _ROOT / "测试" / "测试截图"

WORK_ORDER = "人工智能NLP-RAG-基于Graph RAG 实现金融问答"


def load_json(path: Path, default: Any = None) -> Any:
    """读 JSON（缺失/损坏返回默认值，不抛异常 —— 由用例自己决定是否失败）。"""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return default


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """读 JSONL（缺失返回空列表）。"""
    rows: list[dict[str, Any]] = []
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    except Exception:  # noqa: BLE001
        return []
    return rows


@pytest.fixture(scope="session")
def cfg() -> Any:
    """工单8 配置（阈值全部来自配置层）。"""
    from app.graphrag.config import get_config

    return get_config()


@pytest.fixture(scope="session")
def graph() -> Any:
    """内存图谱（会话级缓存；缺失则跳过依赖它的用例）。"""
    from app.graphrag.config import get_config
    from app.graphrag.store import load_graph

    store = load_graph(get_config())
    if not store.ready:
        pytest.skip(f"图谱不可用（{GRAPH_DIR}）—— 请先跑 研发/scripts/build_graph.py")
    return store


@pytest.fixture(scope="session")
def engine() -> Any:
    """Graph RAG 引擎（**不预热生成模型**，检索用例不需要 Ollama）。"""
    from app.graphrag.engine import GraphRAGEngine

    return GraphRAGEngine()


@pytest.fixture(scope="session")
def question_set() -> list[dict[str, Any]]:
    """17 题测试问题集。"""
    rows = load_jsonl(_ROOT / "研发" / "data" / "测试问题集.jsonl")
    if not rows:
        pytest.skip("测试问题集缺失（研发/data/测试问题集.jsonl）")
    return rows
