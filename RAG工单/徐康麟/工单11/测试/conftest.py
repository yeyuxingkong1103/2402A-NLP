# 工单编号：人工智能NLP-RAG 项目-Embedding 模型微调任务
# -*- coding: utf-8 -*-
"""pytest 公共配置：把 研发/src 加入 import 路径，并提供路径夹具。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "研发" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture(scope="session")
def 仓库根() -> Path:
    return REPO


@pytest.fixture(scope="session")
def 数据集目录(仓库根: Path) -> Path:
    return 仓库根 / "研发" / "data" / "dataset"


@pytest.fixture(scope="session")
def 数据集统计(数据集目录: Path) -> dict:
    p = 数据集目录 / "数据集统计.json"
    if not p.exists():
        pytest.skip(f"缺少数据集统计文件：{p}（请先运行 研发/src/generate_dataset.py）")
    return json.loads(p.read_text(encoding="utf-8"))


def 读jsonl(path: Path, 上限: int | None = None) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if 上限 and len(rows) >= 上限:
                break
    return rows


@pytest.fixture(scope="session")
def 读jsonl函数():
    return 读jsonl
