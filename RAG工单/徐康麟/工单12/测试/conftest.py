# 工单编号：人工智能NLP-RAG 项目-LightRAG 优化
"""pytest 公共夹具（工单12 / t4 测试专用）。

要点：
  * 把 `研发/` 加入 sys.path，用例可直接 `from lightrag import ...`；
  * 把 `tmp_path` 重定向到 **工单目录内** 的 `测试/临时/pytest/<用例名>/`，
    避开系统临时目录在某些沙箱下的权限问题（共享约定 §2 运行姿势）；
  * 所有用例均离线（不调用 Ollama），秒级完成，可在收工前三查里复跑。
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DEV_ROOT = REPO_ROOT / "研发"
TMP_ROOT = REPO_ROOT / "测试" / "临时" / "pytest"

if str(DEV_ROOT) not in sys.path:
    sys.path.insert(0, str(DEV_ROOT))


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """工单12 根目录。"""
    return REPO_ROOT


@pytest.fixture
def tmp_path(request) -> Path:
    """把 tmp_path 钉到工单内的 `测试/临时/pytest/<用例名>/`。"""
    safe = "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in request.node.name)
    path = TMP_ROOT / safe
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# 产物夹具（session 级，只读真实落盘文件，不做任何写操作）
# ---------------------------------------------------------------------------
def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


@pytest.fixture(scope="session")
def cfg():
    """生效配置（内置默认 ← 部署/配置/lightrag.json ← 环境变量）。"""
    from lightrag.config import load_config

    return load_config()


@pytest.fixture(scope="session")
def graph_nodes() -> list[dict]:
    """图谱节点表（研发/data/graph/nodes.jsonl）。"""
    return _read_jsonl(REPO_ROOT / "研发" / "data" / "graph" / "nodes.jsonl")


@pytest.fixture(scope="session")
def graph_edges() -> list[dict]:
    """图谱关系表（研发/data/graph/edges.jsonl）。"""
    return _read_jsonl(REPO_ROOT / "研发" / "data" / "graph" / "edges.jsonl")


@pytest.fixture(scope="session")
def chunks_by_id() -> dict[str, dict]:
    """块表（只留两份去水印文档，与检索侧 DEFAULT_DOC_FILTER 口径一致）。"""
    path = REPO_ROOT / "研发" / "data" / "processed" / "chunks.jsonl"
    out: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if str(row.get("doc_key")) in ("招股说明书1无水", "招股说明书2pd"):
                out[str(row["chunk_id"])] = row
    return out
