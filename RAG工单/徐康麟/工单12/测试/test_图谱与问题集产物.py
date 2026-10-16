# 工单编号：人工智能NLP-RAG 项目-LightRAG 优化
"""t4 用例：图谱产物 + 16 题问题集（读真实落盘文件，离线）。

对应验收点：设计\\验收标准.md 1.3/1.4（类型合法、头尾相容）、2.1（16 题一致）、
P2.1/P2.2/P2.3（图谱落盘与可视化）、R5（只跑小样本，禁止全量）。
"""
from __future__ import annotations

import json
from pathlib import Path

from lightrag import entity_schema as es

REPO_ROOT = Path(__file__).resolve().parents[1]
GRAPH_DIR = REPO_ROOT / "研发" / "data" / "graph"
QUESTION_FILE = REPO_ROOT / "研发" / "评估" / "测试问题集.jsonl"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

# 工单原文 16 题题号（设计\验收标准.md 2.1）
EXPECTED_IDS = {1, 2, 3, 4, 5, 6, 33, 34, 95, 207, 260, 531, 543, 793, 795, 957}

NODE_FIELDS = {"id", "name", "type", "freq", "degree", "pages", "doc_keys"}
EDGE_FIELDS = {"head", "rel", "tail", "weight", "source_chunks", "pages"}


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def test_图谱_节点文件可读且字段类型合法():
    nodes = _read_jsonl(GRAPH_DIR / "nodes.jsonl")
    assert nodes, "nodes.jsonl 为空"
    for node in nodes:
        missing = NODE_FIELDS - set(node)
        assert not missing, f"节点缺字段 {sorted(missing)}：{node.get('name')}"
        assert str(node["name"]).strip()
        assert node["type"] in es.ENTITY_TYPES, f"非法实体类型：{node['type']}"


def test_图谱_关系文件可读且端点可追溯():
    node_rows = _read_jsonl(GRAPH_DIR / "nodes.jsonl")
    # 节点 id 经 _slug_id 去括号（如「保荐机构(主承销商)」→「保荐机构主承销商」），
    # 边按 name 引用，故端点校验同时接受 id 与 name 两种写法。
    endpoints = {n["id"] for n in node_rows} | {n["name"] for n in node_rows}
    edges = _read_jsonl(GRAPH_DIR / "edges.jsonl")
    assert edges, "edges.jsonl 为空"
    for edge in edges:
        missing = EDGE_FIELDS - set(edge)
        assert not missing, f"边缺字段 {sorted(missing)}：{edge}"
        assert edge["rel"] in es.RELATION_TYPES, f"非法关系类型：{edge['rel']}"
        assert edge["head"] in endpoints and edge["tail"] in endpoints, (
            f"边端点不在节点表：{edge['head']} -{edge['rel']}-> {edge['tail']}")


def test_图谱_统计数与落盘行数一致():
    stats = json.loads((GRAPH_DIR / "graph_stats.json").read_text(encoding="utf-8"))
    assert stats["nodes"] == len(_read_jsonl(GRAPH_DIR / "nodes.jsonl"))
    assert stats["edges"] == len(_read_jsonl(GRAPH_DIR / "edges.jsonl"))
    assert stats["entity_types_declared"] == len(es.ENTITY_TYPES)
    assert stats["relation_types_declared"] == len(es.RELATION_TYPES)


def test_图谱_抽样口径合规且可视化产物真实存在():
    meta = json.loads((GRAPH_DIR / "graph_meta.json").read_text(encoding="utf-8"))
    sampling = meta["sampling"]
    assert 0 < sampling["selected"] <= 400, "抽样量越界（R5：≤400 块，禁止全量 3013）"
    assert sampling["selected"] < sampling["chunk_total"], "抽样量不得等于全量"
    assert sampling["sample_seed"], "缺少随机种子，抽样不可复现"
    assert "抽样口径" in meta
    assert (GRAPH_DIR / "graph.graphml").is_file()
    viz = REPO_ROOT / "测试" / "测试截图" / "知识图谱可视化.png"
    assert viz.is_file(), f"缺少图谱可视化：{viz}"
    assert viz.read_bytes()[:8] == PNG_MAGIC, "可视化产物不是合法 PNG"


def test_问题集_16题且题号与工单原文一致():
    rows = _read_jsonl(QUESTION_FILE)
    assert len(rows) == 16, f"题数应为 16，实际 {len(rows)}"
    assert {r["id"] for r in rows} == EXPECTED_IDS
    for row in rows:
        assert str(row["question"]).strip()
        assert str(row["reference_answer"]).strip()
        assert row["expected_pages"], f"id {row['id']} 缺 expected_pages，命中不可追溯"
